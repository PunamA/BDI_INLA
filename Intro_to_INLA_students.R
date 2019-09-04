rm(list=ls())
library(INLA)
library(malariaAtlas)
library(raster)
library(sp)
library(tidyverse)
library(lattice)     
library(gridExtra)

# Bring in Data -----------------------------------------------------------
#for data prep please see the data_prep.R script
load('input/MDG_clean.Rdata')

# Mesh Construction -------------------------------------------------------
#Disclaimer
#there is no rule to determine the right size and spatial extension of the mesh. It is up to the analyst to set the mesh parameters, which vary from case to case. Models based on mesh with a large number of vertices are more computationally demanding and may not necessarily lead to better results than coarser mesh. Therefore we recommend to use a relatively coarse mesh during the preliminary phase of the analysis and use a finer grid only as final step, when the results of the analysis are satisfactory. To get more details on how to build a mesh, see Section 2.1. of Lindgren and Rue paper: https://www.stat.washington.edu/peter/591/Lindgren.pdf.

#the mesh is a trade off between the random field and the computational cost
#from our understanding, the finer the mesh (i.e. mesh$n > number of point) then it will be
#computationally expensive; unfortunately, getting the right mesh takes a bit of trial and error
#in my opinion, have mesh$n approximately be similar or less than number of points would be better

## x and y coordinates in the Response data
coords = cbind(MDG_pr_data$longitude, MDG_pr_data$latitude)
bdry <- inla.sp2segment(MDG_shp)
bdry$loc <- inla.mesh.map(bdry$loc)

#using max edge
#max.edge is the largest allowed triangle length; the lower the number the higher the resolution
#max.edge = c(inside the boundary triangle, outside the boundary triangle); this is useful to avoid

#with max edge only
mesh0 <- inla.mesh.2d(loc = coords, boundary = bdry, max.edge=c(0.5))
par(mfrow=c(1,1))
plot(MDG_shp)
plot(mesh0,add=TRUE)

#max edge with different inner and outer edge values (for boundary effect)
mesh1 <- 
par(mfrow=c(1,1))
plot(MDG_shp)
plot(mesh1,add=TRUE)

#adding offset
#offset is defining how far you want to extend your domain (i.e. a secondary boundary box) 
#for inner boundary and outer boundary; the offset goes with the max edge it should be 
#in the same geographical unit as the max.edge

#NOTE: including boundary argument makes the inner boundary value redundant.
#you can try with removing the boundary at this point and you'll see a different inner
#and outer edge
mesh2 <- 
par(mfrow=c(1,1))
plot(MDG_shp)
plot(mesh2,add=TRUE)

#cutoff can be used to avoid building too many small triangles around clustered data locations
mesh3 <- 
par(mfrow=c(1,1))
plot(MDG_shp)
plot(mesh3,add=TRUE)


#QUESTION 1: play with different values for max.edge, offset and cutoff
#and see what it looks like

# Spatial Construction -----------------------------------------------
#Spatial models can be thought of as multivariate gaussian model;
#the correlation structure is determined by a spatial covariance function (modelled as a Matern covariance function)
#for computational reasons, the full Gaussian field is approximated by a Gaussian Markov Random Field (GMRF)
#To approximte the GMRF is very computationally intense which is why you use INLA

#build A
#the A matrix maps the Gaussian Markov Random Field (GMRF) from the mesh nodes to the n observation location
# you end up with a matrix on number of observations by number of nodes

A<-
dim(A)

#create spde for spatial structure
#this is a function that build upon the spatial structure we assume exists on our mesh
#it is the equivalent of using a GMRF in a spatial model.

spde <- 

#Note that the range and variance parameters need to be interpreted 
#with caution as they are not fully identifiable by the fitted model.

#lastly, we create all the required indexes for the SPDE model
#in here we define the "name of the effect" which will be used in the formula
iset <- 

# Data Stacking -----------------------------------------------------------
#create an INLA stack with the input data
#this is the INLA structure. Think of it as creating a list of the inputs
#you input the data (i.e. your response)
# then the A matrix and you add a 1 for the list of covariates
#the effects should contain a list of the spatial effects and list of covariates

stk <- 
  
  
  
  
  

# Model Building ----------------------------------------------------------
#this is similar to glm/glmer; create a formula with spatial field

#you add +1 for the intercept/ -1 if wish to exclude 
formula0<-y ~ +1 + Elevation + f(spatial.field, model=spde) 

#1. Fit the model (the INLA function)
model0<-inla()

#model checks
summary(model0)


# Model Selection ---------------------------------------------------------

###model selection with WAIC (other criteria can be used)
mypb <- txtProgressBar(min = 0, max = 5, initial = 0, width = 150, style = 3) 

for(i in 1:5){
  
  f1 <- as.formula(paste0("y ~ +1 + f(spatial.field, model=spde) + ", paste0(colnames(covs_df)[1:i], collapse = " + ")))
  
  model1<-inla(f1, data=inla.stack.data(stk,spde=spde),family= 'binomial', 
               Ntrials = n,
               control.predictor=list(A=inla.stack.A(stk),compute=TRUE),
               control.compute = list(dic = TRUE, cpo=TRUE, waic = TRUE)) #verbose=TRUE,
  
  model_selection <- if(i==1){rbind(c(model = paste(colnames(covs_df)[1:i]),waic = model1$waic$waic))}else{rbind(model_selection,c(model = paste(colnames(covs_df)[1:i],collapse = " + "),waic = model1$waic$waic))
  }
  setTxtProgressBar(mypb, i, title = "number complete", label = i)
  
}

model_selection

#QUESTION 2: which model do you think is the best? Why?


# Model Fit (Best Model) --------------------------------------------------

#refit for best model:
formula<-y ~ +1 + f(spatial.field, model=spde) + Access + Elevation + EVI + LST_day

model1<-inla(formula, data=inla.stack.data(stk,spde=spde),family= 'binomial', 
             Ntrials = n,
             control.predictor=list(A=inla.stack.A(stk),compute=TRUE),  #compute gives you the marginals of the linear predictor
             control.compute = list(dic = TRUE, waic = TRUE, config = TRUE), #model diagnostics and config = TRUE gives you the GMRF
             verbose = FALSE) #can include verbose=TRUE to see the log

#model checks
summary(model1)

#you can look at specifics (fixed parameters)




#Get the spatial parameters (estimated) - (convert theta1 and theta2) - to Matern parameters:
# as in the Matern function: Theta1 = log(tau); Theta2 = log(kappa) [see relation ]
model1$summary.hyperpar
model1.res<-inla.spde2.result(model1, 'spatial.field', spde, do.transf=TRUE) #and then look at 
model1.res$summary.log.range.nominal
model1.res$summary.log.variance.nominal


#Plot the Estimated Parameters
##observe the plots for fixed parameters
par(mfrow=c(3,2))
plot(model1$marginals.fixed[[1]], ty = "l", xlab = expression(beta[0]), ylab = "Density") 
plot(model1$marginals.fixed[[2]], ty = "l", xlab = expression(beta[Access]), ylab = "Density") 
plot(model1$marginals.fixed[[3]], ty = "l", xlab = expression(beta[Elevation]), ylab = "Density") 
plot(model1$marginals.fixed[[4]], ty = "l", xlab = expression(beta[EVI]), ylab = "Density") 
plot(model1$marginals.fixed[[5]], ty = "l", xlab = expression(beta[LST_day]), ylab = "Density") 

#observe the plots for hyper parameters
par(mfrow=c(1,3))
plot(model1.res$marginals.var[[1]], ty = "l", xlab = expression(sigma[randomfield]^2), ylab = "Density") 
plot(model1.res$marginals.kap[[1]], type = "l", xlab = expression(kappa), ylab = "Density")
plot(model1.res$marginals.range[[1]], type = "l", xlab = "range nominal", ylab = "Density")

#Project on a grid to look at the random field
#looking at the spatial field and what it looks like
gproj <- inla.mesh.projector(mesh3,  dims = c(300, 300))
g.mean <- inla.mesh.project(gproj, model1$summary.random$spatial.field$mean)
g.sd <- inla.mesh.project(gproj, model1$summary.random$spatial.field$sd)

grid.arrange(levelplot(g.mean, scales=list(draw=F), xlab='', ylab='', main='mean',col.regions = heat.colors(16)),
             levelplot(g.sd, scal=list(draw=F), xla='', yla='', main='sd' ,col.regions = heat.colors(16)), nrow=1)


# Model Prediction (OPTION1)--------------------------------------------------------
### OPTION 1: Prediction combined with fitting 

#create a grid surface with coordinates for each pixel
reference.image <- raster('covariates/Access.tif')
in.country <- which(!is.na(getValues(reference.image)))
reference.coordinates <- coordinates(reference.image)[in.country,]

#make these into points and extract covariates for prediction grid
pred.points <- SpatialPoints(reference.coordinates, proj4string = crs(MDG_shp))
covs <- list.files('covariates/', full.names = T) %>% stack()
pred.covs <- raster::extract(covs, pred.points, df=T)

#remake the A matrix for prediction
Aprediction <- inla.spde.make.A(mesh =         , loc =               );
dim(Aprediction)

#remake the stack for prediction
stk.pred <- inla.stack(data=list(y=NA), #the response
                       A=list(Aprediction,1),  #the A matrix
                       #these are your covariates and spatial components
                       effects=list(iset,
                                    list(Elevation = pred.covs$Elevation,
                                         Access=pred.covs$Access,
                                         LST_day = pred.covs$LST_day,
                                         Rain = pred.covs$Rain,
                                         EVI = pred.covs$EVI)
                       ), 
                       #this is a quick name so you can call upon easily
                       tag='pred')
#join the prediction stack with the one for the full data
stk.full <- inla.stack(stk, stk.pred)

#run an inla model for the prediction 
##NOTE: since this code can take several minutes; 
#we've run it and provided the output for you to see 
# p.res.pred<-inla(formula, data=inla.stack.data(stk.full,spde=spde),family= 'binomial', 
#              Ntrials = n,
#              control.predictor=list(link = 1, A=inla.stack.A(stk.full),compute=FALSE),  #compute gives you the marginals of the linear predictor
#              control.compute = list(config = TRUE), #model diagnostics and config = TRUE gives you the GMRF
#              control.inla(strategy = 'simplified.laplace', huge = TRUE),  #this is to make it run faster
#             verbose = FALSE) #can include verbose=TRUE to see the log

#This will still take time if you'd like to get a cup of tea :) 
download.file(url="https://storage.googleapis.com/map-bdi-spatial-analysis-day/predictionINLA.Rdata", destfile="output/predictionINLA.Rdata")
load('output/predictionINLA.Rdata')

## Extracting Predicted Values
index.pred<-inla.stack.index(stk.full, "pred")$data
post.mean.pred.logit<-p.res.pred$summary.linear.predictor[index.pred,"mean"]
p.pred<-exp(post.mean.pred.logit)/(1 + exp(post.mean.pred.logit))

###visualisation
x <- as.matrix(reference.coordinates)
z <- as.matrix(p.pred)
pr.mdg.in<-rasterize(x, reference.image, field=z, fun='last', background=0)
par(mfrow=c(1,1))
plot(pr.mdg.in)
writeRaster(pr.mdg.in, filename="output/PR.MDG_withinINLA.tif",
            format = "GTiff", overwrite=TRUE, options = c('COMPRESS' = 'LZW'))

# Model Prediction (OPTION2)--------------------------------------------------------
### OPTION 2: Prediction after fitting
## using results from Model1
model = model1
## recall:: formula<-y ~ +1 + f(spatial.field, model=spde) + Access + Elevation + EVI + LST_day

# Covariates for prediction points
Access<- pred.covs$Access
Elevation <-  pred.covs$Elevation
EVI <- pred.covs$EVI
LST_day <-  pred.covs$LST_day

#create the spatial structure
sfield_nodes <- model$summary.random$spatial.field['mean']
field <- (Aprediction %*% as.data.frame(sfield_nodes)[, 1])

#make empty matrix to fill predictions
pred <- matrix(NA, nrow = dim(Aprediction)[1], ncol = 1)

## Calculate Predicted values using regression formula
pred <- model$summary.fixed['(Intercept)', 'mean'] + 
  model$summary.fixed['Access', 'mean'] * Access +
  model$summary.fixed['Elevation', 'mean'] * Elevation +
  model$summary.fixed['EVI', 'mean'] * EVI +
  model$summary.fixed['LST_day', 'mean'] * LST_day +
  field 

# write results in csv
results <- exp(pred)/(1+exp(pred))

# write results as a raster
x <- as.matrix(reference.coordinates)
z <- as.matrix(results)
pr.mdg.out <- rasterFromXYZ(cbind(x, z))

plot(pr.mdg.out)

writeRaster(pr.mdg.out, filename="output/PR.MDG_outsideINLA.tif",
            format = "GTiff", overwrite=TRUE, options = c('COMPRESS' = 'LZW'))

## Optional - Depends on Time Availability  
# Model Validation ---------------------------------------------------
#First we will split the data into training and testing sets (75% training)
## 75% of the sample size
smp_size <- floor(0.75 * nrow(MDG_pr_data))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(MDG_pr_data)), size = smp_size, replace = FALSE)

train <- MDG_pr_data[train_ind, ]
test <- MDG_pr_data[-train_ind, ]
test$positive <- NA  #make the y values for test NA

#Next we rebuild the A matrix for training
train_coords <- coords[train_ind,]
Ae<-inla.spde.make.A(mesh=mesh3,loc=as.matrix(train_coords));dim(Ae)

#we can also build one for the testing (i.e. predictions)
test_coords <- coords[-train_ind,]
Ap <- inla.spde.make.A(mesh = mesh3, loc = test_coords);dim(Ap)

#Plot the train and test sets
par(mfrow=c(1,2))
plot(MDG_shp)
points(test_coords, pch=21, bg=1,col="white", cex=1.2)
plot(MDG_shp)
points(train_coords, pch=21, bg=1,col="blue", cex=1.2)

#Next we create the stacks for estimates and predictions
stk.e <- inla.stack(data=list(y=train$positive, n=train$examined), #the response
                    A=list(Ae,1),  #the A matrix; the 1 is included to make the list(covariates)
                    #these are your covariates
                    effects=list(c(list(intercept=1), #check specification of the intercept
                                   inla.spde.make.index("spatial.field", spde$n.spde)),  #the spatial index
                                 list(Elevation = train$Elevation,
                                      Access=train$Access,
                                      LST_day = train$LST_day,
                                      Rain = train$Rain,
                                      EVI = train$EVI)
                    ), 
                    #this is a quick name so you can call upon easily
                    tag='est')

stk.p <- inla.stack(data=list(y=test$positive, n=test$examined), #the response
                    A=list(Ap,1),  #the A matrix; the 1 is included to make the list(covariates)
                    #these are your covariates
                    effects=list(c(list(intercept=1), #check specification of the intercept
                                   inla.spde.make.index("spatial.field", spde$n.spde)),  #the spatial index
                                 list(Elevation = test$Elevation,
                                      Access=test$Access,
                                      LST_day = test$LST_day,
                                      Rain = test$Rain,
                                      EVI = test$EVI)
                    ), 
                    #this is a quick name so you can call upon easily
                    tag='pred')
#put them together
stk.full <- inla.stack(stk.e, stk.p)

p.res<-inla(formula, data=inla.stack.data(stk.full,spde=spde),family= 'binomial', 
            Ntrials = n,
            control.predictor=list(link = 1, A=inla.stack.A(stk.full),compute=TRUE),  #compute gives you the marginals of the linear predictor
            control.compute = list(config = TRUE), #model diagnostics and config = TRUE gives you the GMRF
            verbose = FALSE) #can include verbose=TRUE to see the log

#getting the predictions
index.pred <- inla.stack.index(stk.full, "pred")$data
post.mean.logit <- p.res$summary.linear.predictor[index.pred,'mean'] #the posterior is in logit form
pred <- exp(post.mean.logit)/(1 + exp(post.mean.logit))
obs <- test$pr #this is the number pos/number examined

#plot to see
plot(obs, pred, xlab = "Observed", ylab = "Predicted", col='red')
abline(a=0, b=1)
cor(obs, pred)  #not great :(

#prediction not good - Not sure why its bad,
#maybe the empirical logit transform would help? 
#Or maybe just need nonlinear covariates.
