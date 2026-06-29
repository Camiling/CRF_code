source('functions/help_funcs/convert_to_rpart.R')

plot_custom_tree <- function(custom_tree, data, target_col) {
  # Plot binary decision tree created by `CTree`
  # Arguments:
  # - custom_tree: Object returned by `CTree`
  # - data: Dataset used to generate the tree
  # - target_col: Name of the target variable
  
  # First convert to rpart format
  rpart_tree <- convert_to_rpart(custom_tree, data, target_col)
  # Plot using rpart plotting function
  rpart.plot(rpart_tree)
}
