import argparse
import json
import pandas as pd

from rpy2.robjects import r
from rpy2.robjects import pandas2ri

pandas2ri.activate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--covariates", default="")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    covariates = [c.strip() for c in args.covariates.split(",") if c.strip()]

    if len(covariates) == 0:
        raise ValueError("binary_edrip requires at least one covariate.")

    required_cols = [args.treatment, args.outcome] + covariates
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Keep only requested columns and drop missing rows
    df = df[required_cols].dropna().copy()

    # Internal generic names:
    # T = treatment, Y = binary outcome, X1..Xd = covariates
    new_cols = ["T", "Y"] + [f"X{i+1}" for i in range(len(covariates))]
    df.columns = new_cols

    r_df = pandas2ri.py2rpy(df)
    r.assign("df", r_df)

    r(
        r'''
        suppressPackageStartupMessages({
          library(dplyr)
          library(purrr)
        })

        find_pi <- function(hx_pi, hy_pi, cutoff) {
          if (cutoff >= hx_pi[length(hx_pi)]) {
            pi <- hy_pi[length(hy_pi)]
          } else if (cutoff <= hx_pi[1]) {
            pi <- hy_pi[1]
          } else {
            row <- sum(cutoff >= hx_pi)
            weight1 <- (hx_pi[row + 1] - cutoff) / (hx_pi[row + 1] - hx_pi[row])
            weight2 <- (cutoff - hx_pi[row]) / (hx_pi[row + 1] - hx_pi[row])
            pi <- weight1 * hy_pi[row] + weight2 * hy_pi[row + 1]
          }
          pi <- max(0.00001, min(0.99999, pi))
          return(pi)
        }

        log_likelihood <- function(params, x, ts, hx_pi, hy_pi, y = c()) {
          is_propensity_model <- length(y) == 0

          calculate_log_likelihood <- function(row) {
            cutoff <- as.numeric(x[row, ] %*% params)
            val <- find_pi(hx_pi, hy_pi, cutoff)
            t <- ts[row]

            if (is_propensity_model) {
              lnL <- t * log(val) + (1 - t) * log(1 - val)
            } else {
              y_val <- y[row]
              lnL <- y_val * log(val) + (1 - y_val) * log(1 - val)
            }
            lnL
          }

          total_log_likelihood <- -sum(sapply(1:nrow(x), calculate_log_likelihood))
          return(total_log_likelihood)
        }

        propensity_model <- function(r, db_double, x, t, ita = 0.001) {
          i <- 1
          repeat {
            kk <- db_double %>%
              mutate(xr = as.matrix(x) %*% r) %>%
              arrange(xr)

            isofit <- isoreg(kk$xr, kk[[t]])
            hx_pi <- kk$xr
            hy_pi <- isofit$yf

            r_opt <- nlminb(
              start = r,
              objective = log_likelihood,
              x = as.matrix(kk[, colnames(x)]),
              ts = kk[[t]],
              hx_pi = hx_pi,
              hy_pi = hy_pi,
              y = c(),
              lower = rep(-5, length(r)),
              upper = rep(5, length(r))
            )

            r_old <- r
            r <- r_opt$par

            denom <- sqrt(sum(r_old^2))
            if (denom < 1e-8) denom <- 1
            diff <- sqrt(sum((r_old - r)^2)) / denom

            if (diff < ita) break
            i <- i + 1
            if (i > 200) break
          }

          return(r)
        }

        outcome_model <- function(beta, db_double, x, t, y, group = 0, ita = 0.001) {
          i <- 1
          repeat {
            kk <- db_double %>%
              mutate(xr = as.matrix(x) %*% beta) %>%
              filter(!!sym(t) == group) %>%
              arrange(xr)

            isofit <- isoreg(kk$xr, kk[[y]])
            hx_pi <- kk$xr
            hy_pi <- isofit$yf

            beta_opt <- nlminb(
              start = beta,
              objective = log_likelihood,
              x = as.matrix(kk[, colnames(x)]),
              ts = kk[[t]],
              hx_pi = hx_pi,
              hy_pi = hy_pi,
              y = kk[[y]]
            )

            beta_old <- beta
            beta <- beta_opt$par

            denom <- sqrt(sum(beta_old^2))
            if (denom < 1e-8) denom <- 1
            diff <- sqrt(sum((beta_old - beta)^2)) / denom

            if (diff < ita) break
            i <- i + 1
            if (i > 200) break
          }

          return(beta)
        }

        edrip <- function(model1, model2, data) {
          all_variables <- all.vars(model2)
          covariates_name <- all.vars(model1[[3]])

          db_double <- data[, c(all_variables)]
          x <- data[, c(covariates_name), drop = FALSE]
          t <- all.vars(model1[[2]])
          y <- all.vars(model2[[2]])

          # propensity init
          logreg1 <- glm(model1, data = db_double, family = "binomial")
          init_p <- summary(logreg1)$coef[-1, 1]

          if (length(init_p) == 1) {
            init_p <- c(init_p)
          }

          opt_r <- propensity_model(init_p, db_double, x, t)

          # outcome init
          logreg2 <- glm(model2, data = db_double, family = "binomial")
          beta <- summary(logreg2)$coef[-1, 1]
          beta <- beta[names(beta) != t]

          if (length(beta) == 1) {
            beta <- c(beta)
          }

          # estimate beta for m0 and m1
          opt_beta_m0 <- outcome_model(beta, db_double, x, t, y, group = 0)
          opt_beta_m1 <- outcome_model(beta, db_double, x, t, y, group = 1)

          # propensity fit
          include_pi <- db_double %>%
            mutate(xr = as.matrix(x) %*% opt_r) %>%
            arrange(xr)

          isofit_pi <- isoreg(include_pi$xr, include_pi[[t]])
          hx_pi <- include_pi$xr
          hy_pi <- isofit_pi$yf

          include_pi <- include_pi %>%
            mutate(
              cutoff = as.matrix(.[, c(covariates_name), drop = FALSE]) %*% opt_r,
              delta_pi = map_dbl(cutoff, ~ find_pi(hx_pi, hy_pi, .x))
            )
  
          # include_m0 / include_m1 are both built from include_pi.

          include_m0 <- include_pi %>%
            mutate(
              xr = as.matrix(.[, c(covariates_name), drop = FALSE]) %*% opt_beta_m0
            ) %>%
            arrange(xr)

          isofit_m0 <- isoreg(include_m0$xr, include_m0[[y]])
          hx_m0 <- include_m0$xr
          hy_m0 <- isofit_m0$yf

          include_m0 <- include_m0 %>%
            mutate(
              cutoff = as.matrix(.[, c(covariates_name), drop = FALSE]) %*% opt_beta_m0,
              delta_m0 = map_dbl(cutoff, ~ find_pi(hx_m0, hy_m0, .x))
            )

          include_m1 <- include_pi %>%
            mutate(
              xr = as.matrix(.[, c(covariates_name), drop = FALSE]) %*% opt_beta_m1 
            ) %>%
            arrange(xr)

          isofit_m1 <- isoreg(include_m1$xr, include_m1[[y]])
          hx_m1 <- include_m1$xr
          hy_m1 <- isofit_m1$yf

          include_m1 <- include_m1 %>%
            mutate(
              cutoff = as.matrix(.[, c(covariates_name), drop = FALSE]) %*% opt_beta_m1,
              delta_m1 = map_dbl(cutoff, ~ find_pi(hx_m1, hy_m1, .x))
            )

          include_m1 <- include_m1 %>%
            mutate(
              delta_m1 = ifelse(delta_m1 > 0.9, 0.9, ifelse(delta_m1 < 0.1, 0.1, delta_m1))
            )

          include_m0 <- include_m0 %>%
            mutate(
              delta_m0 = ifelse(delta_m0 > 0.9, 0.9, ifelse(delta_m0 < 0.1, 0.1, delta_m0))
            )

          db_combined <- cbind(include_m0, delta_m1 = include_m1$delta_m1)

          final_db <- db_combined %>%
            mutate(
              delta_pi = ifelse(delta_pi > 0.9, 0.9, ifelse(delta_pi < 0.1, 0.1, delta_pi)),
              mu1 = ((!!sym(t)) * !!sym(y)) / delta_pi -
                    (((!!sym(t)) - delta_pi) * delta_m1) / delta_pi,
              mu0 = (((1 - !!sym(t)) * !!sym(y)) / (1 - delta_pi)) +
                    (((!!sym(t)) - delta_pi) * delta_m0) / (1 - delta_pi),
              mu = mu1 - mu0
            ) %>%
            summarise(
              mean_pi = mean(delta_pi),
              mean_mu0 = mean(mu0),
              mean_mu1 = mean(mu1),
              odds_after_adjusting = (mean_mu1 / (1 - mean_mu1)) / (mean_mu0 / (1 - mean_mu0)),
              mean_mu = mean(mu),
              sd_bar = sqrt(var(mu) / n()),
              mu_stan = mean_mu / sd_bar,
              p_val = (1 - pnorm(abs(mu_stan))) * 2
            )

          return(final_db)
        }

        covars <- paste0("X", seq_len(ncol(df) - 2))

        form_t <- as.formula(
          paste("T ~", paste(covars, collapse = " + "))
        )

        form_y <- as.formula(
          paste("Y ~ T +", paste(covars, collapse = " + "))
        )

        res <- edrip(
          form_t,
          form_y,
          data = df
        )
        '''
    )

    result = r["res"]

    output = {
        "ate": float(result.rx2("mean_mu")[0]),
        "mu1": float(result.rx2("mean_mu1")[0]),
        "mu0": float(result.rx2("mean_mu0")[0]),
        "odds_ratio": float(result.rx2("odds_after_adjusting")[0]),
        "mean_pi": float(result.rx2("mean_pi")[0]),
        "p_value": float(result.rx2("p_val")[0]),
        "n_complete_cases": int(df.shape[0]),
        "n_covariates": int(len(covariates)),
    }

    print(json.dumps(output))


if __name__ == "__main__":
    main()