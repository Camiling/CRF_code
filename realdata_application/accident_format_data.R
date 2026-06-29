rm(list=ls())

library(dplyr)
library(data.table)

# === Accidents Dataset
# Saves one training dataset from SC, NC, CA, FL, TN, NY, TX, MI, VA, PA, MN, MO, and one test data set from OR. 
# Columns: Start_Lat, Start_Lng, End_Lat, End_Lng (coords, standardized);
# Severity (1–4), y = 1 if Severity >= 3 else 0 (severe accident target);
# Temperature(F), Humidity(%), Pressure(in), Wind_Speed(mph), Visibility(mi) (weather, standardized);
# Hour (0–23), Dow (1=Mon,…,7=Sun), Month (1–12) from Start_Time;
# z = State code (domain variable, plain text);
# Only numeric features standardized; z kept as categorical; long-tail states excluded.
# Skipping rows (observations) with missing data

# https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

# ---- Accidents (Dec 2020) -> clean X, y, z; train=test split by state; standardized ----


accident_csv = "realdata_application/data/US_Accidents_Dec20_Updated.csv"
test_state   = "OR"
use_states   = c("CA","TX","FL","OR","MN","VA","SC","NY","PA","NC","TN","MI","MO")

cols_keep = c(
  "Start_Time","State","Severity","Start_Lat","Start_Lng","Distance(mi)",
  "Temperature(F)","Humidity(%)","Pressure(in)","Visibility(mi)","Wind_Speed(mph)",
  "Weather_Condition","Wind_Direction","Sunrise_Sunset",
  "Amenity","Bump","Crossing","Junction","No_Exit","Railway","Station","Stop",
  "Traffic_Calming","Traffic_Signal"
)

dt = fread(accident_csv, select = cols_keep, na.strings = c("", "NA", "NaN"))
dt = dt[State %in% use_states]
stopifnot(nrow(dt[State == test_state]) > 0)

# timestamps → features
dt[, Start_Time := sub("\\..*$", "", Start_Time)]
dt[, Start_Time := as.POSIXct(Start_Time, format="%Y-%m-%d %H:%M:%S", tz="UTC")]
dt[, Hour  := as.integer(format(Start_Time, "%H"))]
dt[, DOW   := factor(weekdays(Start_Time), levels = c("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"))]
dt[, Month := factor(format(Start_Time, "%m"), levels = sprintf("%02d", 1:12))]

# y and z
dt[, y := as.integer(Severity >= 3)]
dt[, z := State]

# coerce logical-like to factors
bool_cols = c("Amenity","Bump","Crossing","Junction","No_Exit","Railway","Station",
              "Stop","Traffic_Calming","Traffic_Signal","Sunrise_Sunset")
for (cc in intersect(bool_cols, names(dt))) dt[[cc]] <- factor(as.character(dt[[cc]]))

# numeric & categorical sets
num_cols = c("Start_Lat","Start_Lng","Distance(mi)","Temperature(F)","Humidity(%)",
             "Pressure(in)","Visibility(mi)","Wind_Speed(mph)","Hour")
for (cc in intersect(num_cols, names(dt))) suppressWarnings(dt[[cc]] <- as.numeric(dt[[cc]]))
cat_cols = c("Weather_Condition","Wind_Direction","DOW","Month",
             "Sunrise_Sunset","Amenity","Bump","Crossing","Junction","No_Exit",
             "Railway","Station","Stop","Traffic_Calming","Traffic_Signal")

# split
train_dt = dt[State != test_state]
test_dt  = dt[State == test_state]

# one-hot builder (drops rows with any NA in feature set)
make_Xy = function(d) {
  feats = intersect(c(num_cols, cat_cols), names(d))
  keep  = complete.cases(d[, ..feats])
  d2    = d[keep]
  mm    = model.matrix(~ . - 1, data = d2[, ..feats], na.action = na.omit)
  list(X = as.data.frame(mm, check.names = FALSE),
       y = d2$y,
       z = d2$z)
}

tr = make_Xy(train_dt)
te = make_Xy(test_dt)

# align columns (union), fill missing with 0
all_cols = union(colnames(tr$X), colnames(te$X))
add_missing = function(X, cols) {
  miss = setdiff(cols, colnames(X))
  if (length(miss)) X[ , miss] <- 0
  X[ , cols, drop = FALSE]
}
X_train = add_missing(tr$X, all_cols)
X_test  = add_missing(te$X, all_cols)

# standardize (train stats)
means = colMeans(X_train)
sds   = apply(X_train, 2, sd); sds[sds == 0] <- 1
scale_df <- function(X, m, s) as.data.frame(scale(X, center = m[colnames(X)], scale = s[colnames(X)]))
X_train = scale_df(X_train, means, sds)
X_test  = scale_df(X_test,  means, sds)

# bind y, z
accident_train = cbind(X_train, y = tr$y, z = as.character(tr$z))
accident_test  = cbind(X_test,  y = te$y, z = as.character(te$z))

# quick checks
cat(sprintf("accident_train: %d rows, %d features + y,z\n", nrow(accident_train), ncol(accident_train)-2))
cat(sprintf("accident_test : %d rows, %d features + y,z\n", nrow(accident_test),  ncol(accident_test)-2))
stopifnot(!anyNA(accident_train), !anyNA(accident_test))

# Save Data 
save(accident_train, accident_test, file='realdata_application/data/Accident_2020_formatted.RData')

