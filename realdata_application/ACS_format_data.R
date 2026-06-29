rm(list=ls())

# File for creating three clean, preprocessed ACS datasets: Income, Public Coverage, and Mobility, from the 
# WhyShift benchmark. Each row represents one individual, with features such as age, sex, race, education, 
# work status, hours worked, occupation, and household variables. The target `y` is binary: earning > $50 k (income), 
# having public insurance (pubcov), or living in the same home a year ago (mobility). The variable `z` indicates the state (domain),
# used to study distribution shifts. Data come from all states except one held-out for testing (e.g., PR, Puerto Rico) and 
# are already cleaned—no missing values and consistent encoding across states.


# Required helpers -------------------------------

# install.packages(c("reticulate",)) # Must install once
library(reticulate)
library(dplyr)

# Make we're using the Apple Silicon conda env
use_condaenv("r-whyshift", required = TRUE)

# Import whyshift
whyshift = import("whyshift")

bind_xy = function(X, y, feature_names, z_val) {
  df = as.data.frame(py_to_r(X))
  names(df) = py_to_r(feature_names)
  df$y = as.vector(py_to_r(y))
  df$z = z_val
  df
}

# ACS Data -----------------------------------------
# --- All ACS state-like codes (incl. DC, PR) ---
acs_states_all = c(
  "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
  "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
  "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
  "VA","WA","WV","WI","WY","DC","PR"
)

# --- Correct ACS helper (arg order!) ---
# get_data(task, state, need_preprocess, root_dir, year)
get_acs_one = function(task, state, year = 2018,
                       need_preprocess = TRUE,
                       root_dir = "datasets/acs") {
  out = whyshift$get_data(task, state, need_preprocess, root_dir, as.integer(year))
  list(X = out[[1]], y = out[[2]], feat = out[[3]])
}

# --- Safe binder ---
if (!exists("bind_xy")) {
  bind_xy = function(X, y, feature_names, z_val) {
    df = as.data.frame(py_to_r(X))
    names(df) = py_to_r(feature_names)
    df$y = as.vector(py_to_r(y))
    df$z = z_val
    df
  }
}

# =========================
# ACS Income
# =========================
task = "income"    # or "pubcov", "mobility"
year = 2018        # recommended ACS years in this repo are typically within 2014–2018
test_state = "PR"  # hold-out domain

states_train = setdiff(acs_states_all, test_state)

# -----------------------------
# Helper: try-load one state with error capture
# -----------------------------
load_state_df = function(st) {
  tryCatch({
    out = get_acs_one(task = task, state = st, year = year)
    bind_xy(out$X, out$y, out$feat, z_val = st)
  }, error = function(e) {
    message(sprintf("[WARN] Skipping %s: %s", st, conditionMessage(e)))
    # Show Python-side details for debugging if desired:
    # print(reticulate::py_last_error())
    NULL
  })
}

# -----------------------------
# Build TRAIN (skip failing states)
# -----------------------------
train_list = lapply(states_train, load_state_df)
ok_train = !vapply(train_list, is.null, logical(1))
if (!all(ok_train)) {
  bad_states = states_train[!ok_train]
  message("These train states failed and were skipped: ", paste(bad_states, collapse = ", "))
}
train.income = dplyr::bind_rows(train_list[ok_train])

# -----------------------------
# Build TEST (explicitly check PR)
# -----------------------------
test.income = tryCatch({
  out_test = get_acs_one(task = task, state = test_state, year = year)
  bind_xy(out_test$X, out_test$y, out_test$feat, z_val = test_state)
}, error = function(e) {
  message(sprintf("[ERROR] Test state %s failed: %s", test_state, conditionMessage(e)))
  # Uncomment to see Python traceback:
  # print(reticulate::py_last_error())
  NULL
})

# -----------------------------
# Sanity checks
# -----------------------------
if (!is.null(test.income)) {
  print(dplyr::count(train.income, z) |> dplyr::arrange(dplyr::desc(n)))
  print(dplyr::count(test.income,  z))
  cat(sprintf("Train rows: %d | Test rows (%s): %d\n", nrow(train.income), test_state, nrow(test.income)))
} else {
  cat("Test set not built; see warnings above. Consider changing `test_state` or `year`.\n")
} 

# =========================
# ACS Pubcov
# =========================
task = "pubcov"     
year = 2018        # recommended ACS years in this repo are typically within 2014–2018
test_state = "PR"  # hold-out domain

states_train = setdiff(acs_states_all, test_state)

# -----------------------------
# Build TRAIN (skip failing states)
# -----------------------------
train_list = lapply(states_train, load_state_df)
ok_train = !vapply(train_list, is.null, logical(1))
if (!all(ok_train)) {
  bad_states = states_train[!ok_train]
  message("These train states failed and were skipped: ", paste(bad_states, collapse = ", "))
}
train.pubcov = dplyr::bind_rows(train_list[ok_train])

# -----------------------------
# Build TEST (explicitly check PR)
# -----------------------------
test.pubcov = tryCatch({
  out_test = get_acs_one(task = task, state = test_state, year = year)
  bind_xy(out_test$X, out_test$y, out_test$feat, z_val = test_state)
}, error = function(e) {
  message(sprintf("[ERROR] Test state %s failed: %s", test_state, conditionMessage(e)))
  # Uncomment to see Python traceback:
  # print(reticulate::py_last_error())
  NULL
})

# -----------------------------
# Sanity checks
# -----------------------------
if (!is.null(test.pubcov)) {
  print(dplyr::count(train.pubcov, z) |> dplyr::arrange(dplyr::desc(n)))
  print(dplyr::count(test.pubcov,  z))
  cat(sprintf("Train rows: %d | Test rows (%s): %d\n", nrow(train.pubcov), test_state, nrow(test.pubcov)))
} else {
  cat("Test set not built; see warnings above. Consider changing `test_state` or `year`.\n")
} 

# =========================
# ACS Mobility
# =========================
task = "mobility"   # or "pubcov", "mobility"
year = 2018        # recommended ACS years in this repo are typically within 2014–2018
test_state = "PR"  # hold-out domain

states_train = setdiff(acs_states_all, test_state)

# -----------------------------
# Build TRAIN (skip failing states)
# -----------------------------
train_list = lapply(states_train, load_state_df)
ok_train = !vapply(train_list, is.null, logical(1))
if (!all(ok_train)) {
  bad_states = states_train[!ok_train]
  message("These train states failed and were skipped: ", paste(bad_states, collapse = ", "))
}
train.mobility = dplyr::bind_rows(train_list[ok_train])

# -----------------------------
# Build TEST (explicitly check PR)
# -----------------------------
test.mobility = tryCatch({
  out_test = get_acs_one(task = task, state = test_state, year = year)
  bind_xy(out_test$X, out_test$y, out_test$feat, z_val = test_state)
}, error = function(e) {
  message(sprintf("[ERROR] Test state %s failed: %s", test_state, conditionMessage(e)))
  # Uncomment to see Python traceback:
  # print(reticulate::py_last_error())
  NULL
})

# -----------------------------
# Sanity checks
# -----------------------------
if (!is.null(test.mobility)) {
  print(dplyr::count(train.mobility, z) |> dplyr::arrange(dplyr::desc(n)))
  print(dplyr::count(test.mobility,  z))
  cat(sprintf("Train rows: %d | Test rows (%s): %d\n", nrow(train.mobility), test_state, nrow(test.mobility)))
} else {
  cat("Test set not built; see warnings above. Consider changing `test_state` or `year`.\n")
} 




# Save results ---------------
 save(train.income, test.income, train.mobility, test.mobility, train.pubcov, test.pubcov, 
      file='realdata_application/data/ACS_2018_formatted.RData')





