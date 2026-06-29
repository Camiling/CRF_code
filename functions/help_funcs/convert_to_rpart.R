convert_to_rpart <- function(custom_tree, data, target_col) {
  # Arguments:
  # - custom_tree: Custom tree structure
  # - data: Dataset used to generate the tree
  # - target_col: Name of the target variable
  
  # The target variable must be a factor
  if(!is.factor(data[[target_col]])){
    data[[target_col]] = as.factor(data[[target_col]])
  }
  # Initialize components
  frame <- data.frame(
    var = character(),    # Splitting variable or "<leaf>" for leaves
    n = integer(),        # Number of observations at the node
    wt = numeric(),       # Total weights of observations
    dev = numeric(),      # Deviance (placeholder)
    yval = integer(),     # Predicted class
    complexity = numeric(), # Complexity parameter (placeholder)
    ncompete = integer(), # Competing splits (not used here)
    nsurrogate = integer() # Surrogate splits (not used here)
  )
  # Record each split
  splits=matrix(0,1,5)
  colnames(splits)=c('count', 'ncat', 'improve', 'index', 'adj')
  yval2 <- list()  # Stores matrix rows for yval2
  where <- rep(0, nrow(data))  # Tracks the leaf each observation ends up in
  node_counter <- 1  # Counter for unique node IDs
  
  # Map target levels to numeric values
  levels_map <- levels(data[[target_col]])
  num_classes <- length(levels_map)
  class_indices <- setNames(seq_len(num_classes), levels_map)
  
  # Recursive function to build the frame
  build_frame <- function(tree, parent_where, node_number) {
    current_node <- node_counter
    node_counter <<- node_counter + 1
    if (!is.list(tree)) {
      # Leaf node
      predicted_class <- which.max(table(factor(data[[target_col]][parent_where], levels = levels_map)))
      class_counts <- table(factor(data[[target_col]][parent_where], levels = levels_map))
      probabilities <- class_counts / sum(class_counts)
      nodenums.old = rownames(frame)
      frame <<- rbind(frame, data.frame(
        var = "<leaf>",
        n = sum(parent_where),
        wt = sum(parent_where),
        dev = 0,  # Placeholder for deviance
        yval = predicted_class,
        complexity = 0,
        ncompete = 0,
        nsurrogate = 0
      ))
      rownames(frame) <<- c(nodenums.old, node_number)
      yval2_row <- c(
        predicted_class,
        as.numeric(class_counts),
        as.numeric(probabilities),
        sum(parent_where) / nrow(data)  # Node probability
      )
      yval2[[current_node]] <<- yval2_row
      
      # Update where
      where[parent_where] <<- current_node
      return(current_node)
    }
    # Update yval2
    predicted_class <- which.max(table(factor(data[[target_col]][parent_where], levels = levels_map)))
    class_counts <- table(factor(data[[target_col]][parent_where], levels = levels_map))
    probabilities <- class_counts / sum(class_counts)
    yval2_row <- c(
      predicted_class,
      as.numeric(class_counts),
      as.numeric(probabilities),
      sum(parent_where) / nrow(data)  # Node probability
    )
    yval2[[current_node]] <<- yval2_row
    
    # Split node
    split_var <- tree$feature
    threshold <- tree$threshold
    
    # Apply the split

    left_where <- parent_where & (data[[split_var]] < threshold)
    right_where <- parent_where & (data[[split_var]] >= threshold)
    # Record the split
    # 'count', 'ncat', 'improve', 'index', 'adj'
    splits_row <- c(sum(left_where) + sum(right_where), -1, 0, tree$threshold, 0)
    if(current_node>1){
      rownames.tmp = rownames(splits)
      splits <<- rbind(splits, splits_row)
      rownames(splits) <<- c(rownames.tmp, tree$feature)
    }
    else {
      splits[current_node,] <<- splits_row
      rownames(splits) <<- tree$feature
    }
    # Save frame (not affected by split)
    nodenums.old = rownames(frame)
    frame <<- rbind(frame, data.frame(
      var = split_var,
      n = sum(parent_where),
      wt = sum(parent_where),
      dev = 0,  # Placeholder for deviance
      yval = class_indices[which.max(table(data[[target_col]][parent_where]))][1],
      complexity = 0,
      ncompete = 0,
      nsurrogate = 0
    ))
    rownames(frame) <<- c(nodenums.old, node_number)
    # Recur for left and right branches
    build_frame(tree$nodes$left, left_where, node_number*2)
    build_frame(tree$nodes$right, right_where, node_number*2+1)
    return(current_node)
  }
  
  # Start recursive frame building
  build_frame(custom_tree, rep(TRUE, nrow(data)), node_number=1)
  # Combine yval2 into a matrix
  yval2 <- do.call(rbind, yval2)
  colnames(yval2) <- c(
    "fitted",
    paste0("n", seq_len(num_classes)),
    paste0("prob", seq_len(num_classes)),
    "nodeprob"
  )
  # Add yval2 to the frame
  colnames(yval2) = c('V1', 'V2', 'V3', 'V4', 'V5', 'nodeprob')
  frame$yval2 = yval2
  
  # Create the rpart object
  rpart_object <- list(
    frame = frame,
    where = where,
    call = match.call(),
    terms = terms(as.formula(paste(target_col, "~ .")), data = data),
    method = "class", 
    splits=splits
  )
  class(rpart_object) <- "rpart"
  attr(rpart_object, 'ylevels') <- levels(data[[target_col]])
  return(rpart_object)
}
