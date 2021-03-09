####################################################################
###############  INIT                  #############################
####################################################################

suppressMessages(library("data.table"))
suppressMessages(library("dplyr"))
suppressMessages(library("ggplot2"))
suppressMessages(library("gridExtra"))
suppressMessages(library("reshape2"))

data_path <- "C:/Users/frank/OneDrive/Documents/Assignments/CreditDefault.RData"
load(data_path)

####################################################################


#Q1.
#a--------------
dfdefault <- as.matrix(dfdefault)

train_index <- c(1:1800)

#Split the data
data <- list(train = list(X = dfdefault[train_index, -1],
							y = as.matrix(dfdefault[train_index, 1])),
				test = list(X = dfdefault[-train_index, -1],
							y = as.matrix(dfdefault[-train_index, 1])))

TrainData <- dfdefault[train_index, ]
TestData <- dfdefault[-train_index, ]

fit_logistic <- function(X, Y){

	#Add intercept col
	X <- cbind(rep(1, nrow(X)), X)
	colnames(X)[1] <- "(Intercept)"

	#Starts with the moment estimator
	p <- sum(Y) / length(Y)
	beta <- rep(0, ncol(X))
	beta[1] <- -log((1-p)/p)

	n_iter <- 0
	null_dev <- 0
	saturated_dev <- 0

	#Minimize the negative log-lik 
	#Newton-Raphson
	LL_grad_constant <- t(X) %*% Y

	while(TRUE){

		fit <- X %*% beta
		sigm <- (1 + exp(-fit))^(-1)

		LL <- as.numeric(t(Y) %*% log(sigm) + t(1-Y) %*% log(1 - sigm))
		#Record null deviance
		if(n_iter == 0){

			null_dev <- LL
			LL_old <- LL
			is_over <- FALSE

		} else {

			delta <- max(abs(LL/LL_old), abs(LL_old/LL)) - 1
			if(delta/alpha < 10^(-10)){is_over <- TRUE}
			LL_old <- LL

		}

		#X, multiplied by sigma (1 - sigma) row-wise
		weights <- sigm * (1 - sigm)
		swp <- sweep(X, 1, weights, "*")

		#Grad and Hessian
		g <- -t(X) %*% sigm + LL_grad_constant		
		H <- -t(X) %*% swp

		#g <- grad(log_lik, beta)
		#H <- hessian(log_lik, beta)

		#Stops if the last LL increase was small
		if(is_over){break}

		#Random step size U(0.5, 1)
		alpha <- runif(1, min = 0.75, max = 1)
		d_beta <-  alpha * solve(H) %*% g
		d_beta_nrm <- sqrt(sum(d_beta^2))

		beta <- beta - alpha * d_beta	
		n_iter <- n_iter + 1

	}


	#Outputs
	fisher_matrix <- -solve(H)

	coef_tables <- matrix(nrow = length(beta), ncol = 4)
	colnames(coef_tables) <- c("Estimate", "Std. Error", "z value", "Pr(>|z|)")
	rownames(coef_tables) <- rownames(beta)

	#------- 
	coef_tables[, 1] <- beta
	coef_tables[, 2] <- sqrt(diag(fisher_matrix))
	coef_tables[, 3] <- coef_tables[, 1] / coef_tables[, 2]
	coef_tables[, 4] <- 2 * (1 - pnorm(abs(coef_tables[, 3])))

	#------- 
	deviance_tables <- matrix(nrow = 2, ncol = 6)
	colnames(deviance_tables) <- c("Deviance", "Degrees of Freedom", "Pr(> ChiSq)", "AIC", "AICC", "BIC")
	rownames(deviance_tables) <- c("Null", "Residual")

	deviance_tables[1,1] <- 2*(null_dev - saturated_dev)
	deviance_tables[2,1] <- 2*(LL - saturated_dev)
	deviance_tables[1,2] <- nrow(X) - 1
	deviance_tables[2,2] <- nrow(X) - ncol(X)

	deviance_tables[, 3] <- pchisq(deviance_tables[, 1], deviance_tables[, 2])
	deviance_tables[1, 4] <- 2*null_dev - 2
	deviance_tables[2, 4] <- 2*LL - 2*ncol(X)

	deviance_tables[1, 5] <- deviance_tables[1, 4] + 4 / (nrow(X) - ncol(X) - 1)
	deviance_tables[2, 5] <- deviance_tables[2, 4] + 2 * ncol(X) * (1 + ncol(X)) / (nrow(X) - ncol(X) - 1)

	deviance_tables[1, 6] <- 2*LL - log(nrow(X))
	deviance_tables[2, 6] <- 2*LL - log(nrow(X)) * ncol(X) 	
	deviance_tables[, c(1,4,5,6)] <- -deviance_tables[, c(1,4,5,6)]



	pseudo_r_sq <- 1 - deviance_tables[2,1] / deviance_tables[1,1] 


	#------- 
	#Compute deviance residuals
	residuals <- sign(Y - sigm) * sqrt((-2*(Y * log(sigm) + (1-Y) *log(1 - sigm))))


	#------- 
	#Model fitting function, takes a matrix as its input 

	fit_f <- function(x){

		x <- cbind(rep(1, nrow(x)), x)
		colnames(x)[1] <- "(Intercept)"
		p <- (1 + exp(-x %*% beta))^(-1)
		return(p)

	}


	print("Deviance Residuals:", quote = FALSE)
	print(quantile(residuals))

	print("", quote = FALSE)

	print("Coefficients:", quote = FALSE)
	print(coef_tables)

	print("", quote = FALSE)

	print("Deviances:", quote = FALSE)
	print(deviance_tables)

	print("", quote = FALSE)

	print(paste("Saturated model deviance: ", round(saturated_dev, 2), sep = ""), quote = FALSE)

	print("", quote = FALSE)


	print(paste("Number of Newton-Raphson iterations: ", n_iter, sep = ""), quote = FALSE)


	return(list(coef = beta,
					log_lik = LL,
					scores = g,
					fisher_matrix = fisher_matrix,
					hessian = H,
					coef_table = coef_tables,
					deviance_table = deviance_tables,
					deviance_residuals = residuals,
					pseudo_r_sq = pseudo_r_sq,
					n_iter = n_iter,
					saturated_deviance = saturated_dev,
					fit_f = fit_f,
					fitted_probabilities = sigm))

}


logistic_regression <- fit_logistic(data$train$X, data$train$y)

predictions <- list(train = as.data.frame(logistic_regression$fitted_probabilities),
					test = as.data.frame(logistic_regression$fit_f(data$test$X)))

predictions <- lapply(predictions, function(x){colnames(x) <- "P_default"
												return(x)})

g <- function(x){


	labels <- c(paste("Median:", round(median(x$P_default), 3)),
				paste("Mean:", round(mean(x$P_default), 3)))


	ggplot(x, aes(x = P_default)) + geom_histogram(color="grey", binwidth = 0.01) +
									xlab("E[P(Default | X = x)]") +
									ylab("Count") +
									geom_vline(aes(xintercept = mean(P_default), color = "red"), linetype = "dashed", size = 1) +
									geom_vline(aes(xintercept = median(P_default), color = "blue"), linetype = "dashed", size = 1) +
									scale_color_identity(labels = labels, guide = "legend")

}

histograms <- lapply(predictions, g)
histograms$train <- histograms$train + ggtitle("Histogram of estimated default probabilities ; TRAINING SET")
histograms$test <- histograms$test + ggtitle("Histogram of estimated default probabilities ; TEST SET")

#gridExtra::grid.arrange(grobs = histograms, ncol=1, nrow=2)
print(histograms$train)

#b--------------
glm_logistic <- glm(InDefault ~. , as.data.frame(TrainData), family = binomial())
print(summary(glm_logistic))

#c--------------
print(histograms$test)

#d--------------
ROC <- function(p, y, penalties){

	npoints <- round(length(p) / 3)

	index <- order(p)
	p <- as.numeric(p[index])
	y <- as.numeric(y[index])
	points <- seq(1, length(p), length.out = npoints + 1)[-1]
	points <- unique(round(points))

	confusion_mat_long <- matrix(0, nrow = 4, ncol = npoints + 1)
	rownames(confusion_mat_long) <- c("DD", "DN", "NN", "ND")
	colnames(confusion_mat_long) <- as.character(c("0", p[points]))

	#Start with everything classified as a default, i.e.: treshold := p_hat <= 0
	confusion_mat_long[1, 1] <- sum(y)
	confusion_mat_long[4, 1] <- length(y) - confusion_mat_long[1, 1]

	#Move along quantiles, exploiting the fact that p_hat is sorted
	from <- 1
	for(i in 1:npoints){

		to <- points[i]

		y_temp <- y[from:to]
		m <- length(y_temp)

		#Positive variables swapping classes (bad)
		swap_pos <- sum(y_temp)
		#Negative variables swapping classes (good)
		swap_neg <- m - swap_pos

		#From DD to DN (bad)
		confusion_mat_long[2, i+1] <- confusion_mat_long[2, i] + swap_pos
		confusion_mat_long[1, i+1] <- confusion_mat_long[1, i] - swap_pos

		#From ND to NN (good)
		confusion_mat_long[3, i+1] <- confusion_mat_long[3, i] + swap_neg
		confusion_mat_long[4, i+1] <- confusion_mat_long[4, i] - swap_neg

		from <- to + 1

	}

	#Compute the best treshold given some penalty weights
	#I.e. the 1, 20, 50 
	g <- function(x){

		loss <- confusion_mat_long[2, ] * x + confusion_mat_long[4, ]
		out <- matrix(nrow = 4, ncol = 1)

		#The jth column of the matrix is the j-1 th point
		a <- which.min(loss) - 1
		out[1] <- p[points[a]]
		out[2] <- a / npoints 
		out[3] <- loss[a + 1]
		out[4] <- a

		colnames(out) <- as.character(x)
		rownames(out) <- c("Treshold", "Quantile", "Loss", "N")

		return(out)

	}

	optimal_tresholds <- lapply(as.list(penalties), g)

	#Retrieve the confusion matrices
	g <- function(x){

		j <- as.integer(x[length(x)]) + 1
		cm_long <- confusion_mat_long[, j]

		cm <- matrix(cm_long[c(1, 4, 2, 3)], 2, 2)
		rownames(cm) <- c("1_True", "0_True")
		colnames(cm) <- c("1_Pred", "0_Pred")

		stats <- matrix(nrow = 7, ncol = 1)
		colnames(stats) <- c("Value")
		rownames(stats) <- c("Accuracy", "Sensitivity", "Specificity", "Precision", "NPV", "F1", "Kappa")

		stats[1] <- sum(diag(cm)) / length(y)
		stats[c(2,3)] <- diag(cm) / apply(cm, 1, sum)
		stats[c(4,5)] <- diag(cm) / apply(cm, 2, sum)
		stats[6] <- 2 * (stats[2] * stats[4]) / (stats[2] + stats[4])

		#Kappa
		yes <- (sum(cm[1, ]) * sum(cm[, 1])) / length(y)^2
		no <- (sum(cm[2, ]) * sum(cm[, 2])) / length(y)^2

		stats[7] <- (stats[1] - yes - no) / (1 - yes - no)
		stats <- round(stats, 3)

		return(list(CM = cm,
					stats = stats))

	}

	cmats <- lapply(optimal_tresholds, g)

	g <- function(x){

		dummy <- as.matrix(x[-length(x)])
		rownames(dummy) <- rownames(x)[-length(x)]
		return(dummy)

	}

	optimal_tresholds <- lapply(optimal_tresholds, g)
	names(optimal_tresholds) <- as.character(penalties)
	names(cmats) <- as.character(penalties)


	#Compute the ROC curve
	rates <- confusion_mat_long[c(4, 1), ] 
	n_pos <- sum(y)
	rates[1, ] <- rates[1, ] / (length(y) - n_pos)
	rates[2, ] <- rates[2, ] / n_pos

	rates <- as.data.frame(t(rates))
	colnames(rates) <- c("FPR", "TPR")
	index <- order(rates$FPR)
	rates <- rates[order(rates$FPR), ]

	rates$Random <- seq(0, 1, length.out = nrow(rates))

	#Compute AUC
	ROC_table <- matrix(nrow = 3)
	colnames(ROC_table) <- "Value"
	rownames(ROC_table) <- c("AUC", "Var", "Gini")

	dx <- diff(rates$FPR)
	y_m <- (rates$TPR[-1] + rates$TPR[-length(rates$TPR)])/2
	A <- sum(dx * y_m)

	ROC_table[1] <- A
	ROC_table[2] <- (1/A) * (sum(dx * y_m^2) - A^2)
	ROC_table[3] <- 2*ROC_table[1] - 1

	rates <- melt(rates, id = "FPR")

	ROC_plot <- ggplot(rates, aes(x = FPR, y = value, color = variable, linetype = variable)) + geom_line() +
																					scale_color_manual(values = c("black", "red")) +
																					scale_linetype_manual(values=c("solid", "dashed")) +
																					ggtitle("ROC Curve") +
																					ylab("TPR")
	
	return(list(ROC = list(stats = ROC_table,
							plot = ROC_plot),
				Results = list(loss = optimal_tresholds,
								stats = cmats)))

}


classifier <- ROC(logistic_regression$fitted_probabilities, data$train$y, c(1, 20, 50))
print(classifier$Results$stats)

#e--------------

#....


#f--------------

classifier_test <- ROC(logistic_regression$fit_f(data$test$X), data$test$y, c(1, 20, 50))
print(classifier_test$ROC)





####################################################################
###############  INIT                  #############################
####################################################################

suppressMessages(library("data.table"))
suppressMessages(library("dplyr"))
suppressMessages(library("ggplot2"))
suppressMessages(library("gridExtra"))
suppressMessages(library("reshape2"))


data_path <- "C:/Users/frank/OneDrive/Documents/Assignments/LumberData.RData"
load(data_path)


####################################################################


#Q2.
#a--------------

#Fits both LDA and QDA
#Includes fitted values, and a predict method in the output
LDA <- function(data, y_name, QDA = FALSE){

	if(class(data)[1] != "data.table"){

		print("ERROR: data argument must be a data.table object.", quote = FALSE)
		return(NULL)

	}

	if(!(y_name %in% names(data))){

		print(paste("ERROR: ", y_name, " not found.", sep = ""), quote = FALSE)
		return(NULL)

	}

	predictors <- names(data)[-which(names(data) == y_name)]
	types <- unique(sapply(data[, predictors, with = FALSE], class))
	not_allowed <- c("character", "factor")
	if(any(not_allowed %in% types)){

		print(paste("ERROR: factors are not allowed.", sep = ""), quote = FALSE)
		return(NULL)

	}


	#Query mean, n and % with a by = class clause
	class_stats <- data[, lapply(.SD, mean), by = c(y_name), .SDcols = predictors] %>%
						.[, n := unlist(data[, lapply(.SD, length), by = c(y_name), .SDcols = c(y_name)] %>%
													.[, 2])] %>%
						.[, P := lapply(.SD, function(x){x / sum(x)}), .SDcols = c("n")]

	#Build the class covariance matrix
	setkeyv(data, c("WoodType"))
	setkeyv(class_stats, c("WoodType"))

	classes <- as.list(unique(data[, c(y_name), with = FALSE][[1]]))	
	names(classes) <- unique(data[, c(y_name), with = FALSE][[1]])
		
	#Extract the covariance matrices with by = class
	xtract_vcov <- function(x){

		return(var(data[x, predictors, with = FALSE]))

	}

	vcov_mats <- lapply(classes, xtract_vcov)

	#Pool the matrices together
	#I.e.: weighted average with weights(i) = degree of freedom_i = n_i - 1
	#So kind of like when you pool variances for a t-test with unequal sample sizes
	weight_vcov_mat <- function(x){

		df <- as.numeric(class_stats[x, n] - 1)
		return(df * vcov_mats[[x]] / (nrow(data) - length(classes)))

	}

	vcov_mats <- lapply(classes, weight_vcov_mat)

	#Add the weighted matrices to produce the estimated vcov matrix 
	vcov_mat <- vcov_mats[[1]]
	for(j in 2:length(vcov_mats)){vcov_mat <- vcov_mat + vcov_mats[[j]]}

	#Build function to fit data
	class_names <- unlist(classes)
	fit_x <- function(x){

		output <- as.data.table(matrix(0, nrow = nrow(x), ncol = length(classes)))
		names(output) <- class_names

		for(c in class_names){

			#Obtain class mean
			mu <- as.numeric(unlist(class_stats[c, predictors, with = FALSE]))

			#Fit
			x_mat <- as.matrix(x[, predictors, with = FALSE])
			for(j in 1:ncol(x_mat)){x_mat[, j] <- x_mat[, j] - mu[j]}

			#exp{...} part of the multivariate normal
			#If QDA, use the mle within-class estimate of the covariances
			#If LDA, use the pooled mle estimate
			if(QDA){

				rotation_matrix <- solve(vcov_mats[[c]])

			} else {

				rotation_matrix <- solve(vcov_mat)

			}
	
			linear_val <- -0.5 * apply(x_mat, 1, function(x){x %*% rotation_matrix %*% x})
			linear_val <- linear_val + log(class_stats[c, c("P"), with = FALSE][[1]]) 

			#If QDA, multiply by scaling factor too
			if(QDA){

				linear_val <- linear_val - 0.5*log(det(vcov_mats[[c]]))

			}

			output[, (c) := linear_val]

		}

		#Get prediction
		output[, Prediction := class_names[apply(output, 1, which.max)]]

		#Exponentiate to get densities
		output[, (class_names) := lapply(.SD, exp), .SDcols = class_names]

		#Scale by sums 
		p_sum <- apply(output[, (class_names), with = FALSE], 1, sum)
		output[, (class_names) := lapply(.SD, function(x){x / p_sum}), .SDcols = class_names]

		return(output)

	}

	#Fit
	fits <- fit_x(data) %>%
				.[, Real := data[, y_name, with = FALSE]] %>%
				.[, Hit := as.numeric(Prediction == Real)]

	accuracy <- sum(fits$Hit) / nrow(data)

	if(QDA){

		print("Covariance matrices:", quote = FALSE)
		print("", quote = FALSE)
		print(vcov_mats)
		print("", quote = FALSE)

	} else {

		print("Common covariance matrix:", quote = FALSE)
		print("", quote = FALSE)
		print(vcov_mat)
		print("", quote = FALSE)

	}


	print("Means accross classes:", quote = FALSE)
	print("", quote = FALSE)
	print(class_stats)
	print("", quote = FALSE)

	print(paste("In-sample accuracy: ", round(accuracy, 4), sep = ""), quote = FALSE)

	out <- list(data = data,
					response = y_name,
					predictors = predictors,
					classes = class_names,
					covariance_common = vcov_mat,
					covariance_indiv = vcov_mats,
					mean_indiv = class_stats,
					fitted_values = fits,
					accuracy = accuracy,
					predict = fit_x,
					is_QDA = QDA)
	return(out)

}




LumberData <- as.data.table(LumberData)
LumberData_LDA <- LDA(LumberData, "WoodType")

new_obs <- LumberData[1, ]
new_obs$BarkDarkness <- 3
new_obs$LogCurve <- 3

new_obs_p <- LumberData_LDA$predict(new_obs)
print("Prediction for BarkDarkness = LogCurve = 3:", quote = FALSE)
print(new_obs_p)


#b---------

#Automatically plot the linear or quadratic boundaries depending on the object type
#Uses exact conic forms for QDA
plot_LDA_2D <- function(LDA_obj){

	if(length(LDA_obj$predictors) != 2){

		print("Error: number of predictors must be 2.", quote = FALSE)
		return(NULL)

	}

	ranges <- lapply(as.list(c(1,2)), function(x){round(range(LDA_obj$data[, LDA_obj$predictors[x], with = FALSE]))})
	names(ranges) <- LDA_obj$predictors
	ranges <- lapply(ranges, function(x){

								spread <- x[2] - x[1]
								x[1] <- x[1] - spread/3
								x[2] <- x[2] + spread/3

								return(x)

	})

	axis <- lapply(ranges, function(x){seq(x[1], x[2], length.out = 201)})

	points <- expand.grid(axis[[1]], axis[[2]])
	colnames(points) <- names(axis)
	points <- as.data.table(points)

	fits <- LDA_obj$predict(points)
	points[, (LDA_obj$response) := fits$Prediction]

	#Add the sample to the fitted value
	plot_frame <- rbind.data.frame(points, LDA_obj$data)
	names(plot_frame) <- c("x", "y", "Prediction")

	plot_obj <- ggplot(plot_frame, aes(x = x, y = y, color = Prediction)) + geom_point(alpha = 0.5) +
																				scale_color_manual(values = c("yellow3", "darkorange1", "blue")) +
																				xlab(LDA_obj$predictors[1]) +
																				ylab(LDA_obj$predictors[2]) +
																				labs(color = LDA_obj$response)	

	#Code to get the intersecting lines
	#Obtain the equation of the separating hyperplane for each 3c2 = 3 pairwise combinations
	choices <- as.data.table(expand.grid(LDA_obj$classes, LDA_obj$classes))
	choices[, names(choices) := lapply(.SD, as.factor), .SDcols = names(choices)]

	#Remove duplicates from the expand.grid routine by subsetting from the lower diagonal of the matrix of combinations
	choices_numeric <- choices[, lapply(.SD, as.numeric), .SDcols = names(choices)]
	rmv <- choices_numeric[Var1 >= Var2, which = TRUE]
	choices <- choices[-rmv, ]

	#Compute lines
	lines <- function(i){

		#Lines are obtained via solving a simple 2x2 linear equation system
		vals <- c(unlist(t(choices[i])))

		#Retrieve means
		stats <- LDA_obj$mean_indiv[vals]
		mu <- stats[, LDA_obj$predictors, with = FALSE]
		prior <- stats[, c("P"), with = FALSE]
		#Compute vectors
		cov_inv <- solve(LDA_obj$covariance_common)
		u <- t(as.matrix(mu[1] - mu[2]))
		v <- t(as.matrix(mu[1] + mu[2]))
		k <- cov_inv %*% u

		#Constant equal to the dot product between x and k
		c <- 0.5*t(v) %*% cov_inv %*% u - log(prior[1]/ prior[2])

		#Parameters
		out <- matrix(nrow = 2, ncol = 1)
		#Intercept
		out[1] <- as.numeric(c/k[2])
		out[2] <- as.numeric(-k[1]/k[2])

		return(out)

	}

	#Compute quadratic boundaries
	quads <- function(i){

		vals <- c(unlist(t(choices[i])))

		#Retrieve means
		stats <- LDA_obj$mean_indiv[vals]
		mu <- stats[, LDA_obj$predictors, with = FALSE]
		prior <- stats[, c("P"), with = FALSE]	
		
		covs <- lapply(as.list(vals), function(x){LDA_obj$covariance_indiv[[x]]})
		cov_inv <- lapply(covs, solve)
		
		#Compute various parameters
		mu_1 <- t(as.matrix(mu[1]))
		mu_2 <- t(as.matrix(mu[2]))
		mat_diff <- -0.5 * (cov_inv[[1]] - cov_inv[[2]])
		v <- t(mu_1) %*% cov_inv[[1]] - t(mu_2) %*% cov_inv[[2]]
		d <- log(prior[1]) - log(det(covs[[1]])) - 0.5 * t(mu_1) %*% cov_inv[[1]] %*% mu_1
		d <- d - (log(prior[2]) - log(det(covs[[2]])) - 0.5 * t(mu_2) %*% cov_inv[[2]] %*% mu_2)


		#--------------------------------------
		#Big geometry time

		#The eq is: t(x) %*% mat_diff %*% x + t(v) %*% x + d = 0
		#Obtain the centered form of the conic section via the eigenvalues of mat_diff		
		conic_eigen <- eigen(mat_diff)

		#mat_diff = t(M) %*% D %*% M, so let z = Mx so that:
		#t(z) %*% D %*% z + t(v) %*% t(M) %*% z + d = 0
		#And the centre lies at: 2 * t(z) %*% D + t(v) %*% t(M) = 0
		#So you can use centered coordinates which offsets the constant "d" by (1/4) t(v) %*% t(M) %*% 1/D %*% M %*% v
		v_new <- v %*% conic_eigen$vectors
		k <- as.numeric((1/4) * v_new %*% diag(1/conic_eigen$values) %*% t(v_new) - d)		
		centre <- 0.5 * v_new %*% diag(1/conic_eigen$values)			


		#If the eigenvalue signs differ, then it's an hyperbolla or a straight line
		if(sum(sign(conic_eigen$values)) == 0){


			#Compute y(x) from the hyperbolla equation
			#Use centered, orthogonal coordinates

			#(z + c) %*% D %*% (z + c) = k
			#D_11 (x + c_1)^2 + D_22 (y + c_2)^2 = k
			#y + c_2 = +/- sqrt{ k/D_22 - (D_11/D_22) (x + c_1)^2}
			rotation_mat <- conic_eigen$vectors
			
			#parameters
			a <- rep(k, 2) / conic_eigen$values
			c <- sign(a)
			a <- sqrt(abs(a))

			spawn_parabolla <- function(t){

				#Canonical form
				if(c[1] == 1){

					g1 <- as.numeric(a * c(cosh(t), sinh(t))  - centre)
					g2 <- as.numeric(a * c(-cosh(t), sinh(t))  - centre)

				} else {
				#Conjugate hyperbola otherwise

					g1 <- as.numeric(a * c(sinh(t), cosh(t))  - centre)
					g2 <- as.numeric(a * c(sinh(t), -cosh(t))  - centre)		

				}

				values <- matrix(c(g1, g2), 2, 2)

				#Switch back to canonical basis
				values <- t(rotation_mat %*% values)
				colnames(values) <- c("v1", "v2")

				return(as.data.table(values))

			}


			hyperbolla <- lapply(seq(-2, 2, length.out = 3000), spawn_parabolla)
			names(hyperbolla) <- paste("v", c(1:length(hyperbolla)))
			hyperbolla <- bind_rows(hyperbolla)

			hyperbolla <- hyperbolla[v1 %between% ranges[[1]] & v2 %between% ranges[[2]]]
			names(hyperbolla) <- LDA_obj$predictors

			return(hyperbolla)

		} else {


			#Else, it's an ellipse
			radii <- sqrt(rep(k, 2) / conic_eigen$values)

			ellipse <- sapply(seq(0, 2*pi-10^(-10), length.out = 1000), function(theta){radii * c(cos(theta), sin(theta))  - centre})

			#Canonical basis
			ellipse <- t(conic_eigen$vectors %*% ellipse)


			ellipse <- as.data.table(ellipse)
			names(ellipse) <- c("v1", "v2")
			ellipse <- ellipse[v1 %between% ranges[[1]] & v2 %between% ranges[[2]]]
			names(ellipse) <- LDA_obj$predictors

			return(ellipse)


		}


	}

	#Add boundaries
	if(!LDA_obj$is_QDA){

		line_eqs <- sapply(c(1:nrow(choices)), lines)
		rownames(line_eqs) <- c("(Intercept)", "Slope")
		colnames(line_eqs) <- paste("Eq.", c(1:ncol(line_eqs)), sep = "")

		#Compute the intersections for m <= 3
		if(length(LDA_obj$classes) <= 3){

			intersects <- matrix(nrow = 2, ncol = 1)
			intersects[1] <- (line_eqs[1,1] - line_eqs[1,2]) / (line_eqs[2,2] - line_eqs[2,1])
			intersects[2] <- line_eqs[1,1] + line_eqs[2,1] * intersects[1]
			rownames(intersects) <- LDA_obj$predictors
			colnames(intersects) <- "Coordinate"

		} else {

			intersects <- NULL

		}



		#Add lines to the plot
		for(j in 1:ncol(line_eqs)){

			plot_obj <- plot_obj + geom_abline(intercept = line_eqs[1, j], 
												slope = line_eqs[2, j],
												linetype = "dashed", 
												color = "black", 
												size = 0.8)

		}

		plot_obj <- plot_obj + ggtitle("LDA Plot")
		print(plot_obj)
		return(list(plot = plot_obj,
						lines = list(lines = line_eqs,
										variables = choices),
						intersect = intersects))

	} else {

		quad_lines <- lapply(as.list(c(1:nrow(choices))), quads)

		for(q in quad_lines){

			if(nrow(q) > 0){

				names(q) <- c("x", "y")
				plot_obj <- plot_obj + geom_point(data = q, aes(x = x, y = y), 
													color = "black", size = 0.35)

			}

		}

		plot_obj <- plot_obj + ggtitle("QDA Plot")
		print(plot_obj)
		return(list(plot = plot_obj,
						lines = NULL,
						intersect = NULL))

	}


}

LDA_plot <- plot_LDA_2D(LumberData_LDA)


#c--------------

print(LDA_plot$intersect)

#d--------------

LumberData_QDA <- LDA(LumberData, "WoodType", QDA = TRUE)

new_obs_QDA <- LumberData[1, ]
new_obs_QDA$BarkDarkness <- 3
new_obs_QDA$LogCurve <- 3

new_obs_p_QDA <- LumberData_QDA$predict(new_obs)
print("Prediction for BarkDarkness = LogCurve = 3:", quote = FALSE)
print(new_obs_p_QDA)


#e---------

QDA_plot <- plot_LDA_2D(LumberData_QDA)