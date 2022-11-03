# Databricks notebook source
# MAGIC %md %md
# MAGIC # Description: Distributed R: User Defined Functions in Spark
# MAGIC 
# MAGIC ### Understanding UDFs
# MAGIC 
# MAGIC Both `SparkR` and `sparklyr` support user-defined functions (UDFs) in R which allow you to execute arbitrary R code across a cluster.  The advantage here is the ability to distribute the computation of functions included in R's massive ecosystem of 3rd party packages.  In particular, you may want to use a domain-specific package for machine learning or apply a statisical transformation that is not available through the Spark API.  Running in-house custom R libraries on larger data sets would be another place to use this family of functions.
# MAGIC 
# MAGIC How do these functions work?  The R process on the driver has to communicate with R processes on the worker nodes through a series of serialize/deserialize operations through the JVMs.  The following illustration walks through the steps required to run arbitrary R code across a cluster.
# MAGIC 
# MAGIC <img src="https://github.com/kurlare/misc-assets/blob/master/DistributedR_ControlFlow.png?raw=true" width="800" />
# MAGIC 
# MAGIC Looks great, but what's the catch?  
# MAGIC 
# MAGIC * **You have to reason about your program carefully and understand how exactly these functions are being, *ahem*, applied across your cluster.**  
# MAGIC 
# MAGIC * **R processes on worker nodes are ephemeral.  When the function being applied finishes execution the process is shut down and all state is lost.**
# MAGIC 
# MAGIC * **As a result, you have to pass any contextual data and libraries along with your function to each worker for your job to be successful.**
# MAGIC 
# MAGIC * **There is overhead related to creating the R process and ser/de operations in each worker.**  
# MAGIC 
# MAGIC Don't be surprised if using these functions runs slower than expected.  One of the benefits of running distributed R on Databricks is that you can install libraries at the cluster scope.  This makes them available on each worker and you do not have to pay this performance penalty every time you spin up a cluster.
# MAGIC 
# MAGIC In general, use the native Spark APIs as much as possible.  If there is no way to implement your logic except in R you can turn to UDFs and get the job done.  This is echoed [here](https://therinspark.com/distributed.html) by one of the authors of `sparklyr`.

# COMMAND ----------

# MAGIC %md ## Setup
# MAGIC This notebook runs on DBR 8.0, R 4.0, Spark 3.1.1.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distributed `apply`
# MAGIC 
# MAGIC Between `sparklyr` and `SparkR` there are a number of options for how you can distribute your R code across a cluster with Spark.  Functions can be applied to each *group* or each *partition* of a Spark DataFrame, or to a list of elements in R.  In the following table you can see the whole family of distributed `apply` functions:
# MAGIC 
# MAGIC |  Package | Function Name |     Applied To     |   Input  |    Output    |
# MAGIC |:--------:|:-------------:|:------------------:|:--------:|:-----------:|
# MAGIC | sparklyr |  spark_apply  | partition or group | Spark DF |   Spark DF  |
# MAGIC |  SparkR  |     dapply    |      partition     | Spark DF |   Spark DF  |
# MAGIC |  SparkR  | dapplyCollect |      partition     | Spark DF | R dataframe |
# MAGIC |  SparkR  |     gapply    |        group       | Spark DF |   Spark DF  |
# MAGIC |  SparkR  | gapplyCollect |        group       | Spark DF | R dataframe |
# MAGIC |  SparkR  |  spark.lapply |    list element    |   list   |     list    |
# MAGIC 
# MAGIC Later in this notebook we'll work through these functions one by one.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Leveraging Packages in Distributed R
# MAGIC 
# MAGIC As stated above, everything required for your UDF needs to be passed along with it.  On Databricks you can install packages on the cluster and they will automatically be installed on each worker.  This saves you time and gives you two options to use libraries with a UDF on Databricks:
# MAGIC <br><br>
# MAGIC 
# MAGIC * Load the entire library - `library(broom)`
# MAGIC * Reference a specific function from the library namespace - `broom::tidy()`
# MAGIC 
# MAGIC In the examples below we will train a model on each group of `mtcars` with a distinct `cyl` value.  This will be a simple linear model where `mpg` is the dependent variables, and all other variables (except `cyl`) independent.  Furthermore, we use the `broom` package to tidy up the output of our linear model.  `broom` is available in DBR 8.0, so there's no need to install it yourself to run these examples.  The results will be a Spark DataFrame with different coefficients for each group.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Examples

# COMMAND ----------

# MAGIC %md
# MAGIC ##### `spark_apply`
# MAGIC 
# MAGIC For the first example, we'll use **`spark_apply()`**.
# MAGIC 
# MAGIC `spark_apply` takes a Spark DataFrame as input and must return a Spark DataFrame as well.  By default it will execute the function against each partition of the data, but we will change this by specifying a 'group by' column in the function call.  `spark_apply()` will also distribute all of the contents of your local `.libPaths()` to each worker when you call it for the first time unless you set the `packages` parameter to `FALSE`.  For more details see the [Official Documentation](https://spark.rstudio.com/guides/distributed-r/).  
# MAGIC 
# MAGIC **Note:** To get the best performance, we specify the schema of the expected output DataFrame to `spark_apply`.  This is optional, but if we don't supply the schema Spark will need to sample the output to infer it.  This is costly on longer running UDFs.

# COMMAND ----------

# Push mtcars dataset to Spark
sc <- sparklyr::spark_connect(method = "databricks")

mtcars_sdf <- sparklyr::sdf_copy_to(sc, mtcars, overwrite = T)

# Output schema
schema <- list(cyl = "double",
               term = "string",
               estimate = "double",
               std_error = "double",
               statistic = "double",
               p_value = "double")

## Add a new column for each group and return the results
results_sdf <- sparklyr::spark_apply(mtcars_sdf,
                                  group_by = "cyl",
                                  function(e){
                                    # 'e' is a data.frame containing all the rows for each distinct UniqueCarrier
                                    tidymod <- broom::tidy(lm(mpg ~ ., data = e[, -2]))
                                    tidymod
                                  }, 
                                   # Specify schema
                                   columns = schema,
                                   # Do not copy packages to each worker
                                   packages = F)
head(results_sdf, 10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### `gapply` & `dapply`
# MAGIC 
# MAGIC In `SparkR`, there are separate functions depending on whether you want to run R code on each partition of a Spark DataFrame (`dapply`), or each group (`gapply`).  With these functions you **must** supply the schema ahead of time.  In the next example we will recreate the first but use `gapply` instead.

# COMMAND ----------

library(SparkR)

mtcarsDF <- createDataFrame(mtcars)

resultSchema <- structType(
  structField("cyl", "double"),
  structField("term", "string"),
  structField("estimate", "double"),
  structField("std_error", "double"),
  structField("statistic", "double"),
  structField("p_value", "double")
)

modelsDF <- gapply(mtcarsDF, 
                   cols = c("cyl"),
                   function(key, e) {
                     
                     tidymod <- cbind(key, broom::tidy(lm(mpg ~ ., data = e[, -2])))
                     
                   }, resultSchema)

## Display results & class
cat("Result of gapply: \n")
print(head(modelsDF))
cat(c("\nClass of result from gapply():", class(modelsDF)))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### `gapplyCollect`
# MAGIC This function is identical to `gapply` with one exception - it collects the results back to the driver as a R data.frame instead of keeping it distributed across the cluster as a Spark DataFrame.  

# COMMAND ----------

models_df <- gapplyCollect(mtcarsDF, 
                   cols = c("cyl"),
                   function(key, e) {
                     
                     # Add key to output R data.frame
                     tidymod <- broom::tidy(lm(mpg ~ ., data = e[, -2]))
                     tidymod$cyl <- unlist(key)
                     tidymod 
                   })

## Display results & class
cat("Result of gapplyCollect: \n")
print(head(models_df))
cat(c("\nClass of result from gapplyCollect():", class(models_df)))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### `dapply`
# MAGIC 
# MAGIC There's no notion of group in `dapply` - our R function gets applied to each partition in the Spark DataFrame. 
# MAGIC 
# MAGIC In the following example we will isolate the `mpg` column in mtcars and apply a jitter to each of the numeric values.  While this is a simple operation, there is no native support for `jitter()` in SparkR so this is an example of manipulating data in a Spark DataFrame using the R ecosystem. 

# COMMAND ----------

## With dapply() we need to provide a schema for the returning DataFrame
## Passing the wrong type will yield incorrect results (i.e, integer instead of double)
resultSchema <- structType(
  structField("mpg", "double"),
  structField("jittered_mpg", "double")
)

# Subset data to only mpg
mpgDF <- select(mtcarsDF, "mpg")

## the original and transformed columns
jitteredDF <- dapply(mpgDF,
                     function(e) {
                       e <- cbind(e, jitter(e$mpg))},
                     resultSchema)

## Display results & class
cat("Result of dapply: \n")
print(head(jitteredDF))
cat(c("\nClass of result from dapply():", class(jitteredDF)))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### `dapplyCollect`
# MAGIC 
# MAGIC This function takes a Spark DataFrame as input and returns a regular R data.frame.  As such, it does not require a schema to be passed with it.  The catch here is that the collective results of your UDF need to be able to fit into the driver.  
# MAGIC 
# MAGIC Let's execute the same 'jitter' operation as before, but this time we'll index each partition and only return the first row.  This is useful because row indexing in Spark is not typically possible.  For this task we can reuse the `DepDelayDF` created above.

# COMMAND ----------

# Same transformation as before
jittered_df <- dapplyCollect(mpgDF,
                     function(e) {
                       e <- cbind(e, jitter(e$mpg))
                     })

## Display results & class
cat("Result of dapplyCollect: \n")
print(head(jittered_df))
cat(c("\nClass of result from dapplyCollect():", class(jittered_df)))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### `spark.lapply`
# MAGIC 
# MAGIC This final function is also from SparkR.  It accepts a list and then uses Spark to apply R code to each element in the list across the cluster.  As [the docs](https://spark.apache.org/docs/latest/api/R/spark.lapply.html) state, it is conceptually similar to `lapply` in base R, so it will return a **list** back to the driver.  
# MAGIC 
# MAGIC For this example we'll take a list of strings and manipulate them in parallel, somewhat similar to the examples we've seen so far.

# COMMAND ----------

# Create list 
cylinders <- list(4, 6, 8)

list_of_dfs <- spark.lapply(cylinders, 
                            function(e) {
                              cyl_df <- mtcars[mtcars$cyl == e, ]
                              broom::tidy(lm(mpg ~ ., data = cyl_df[, -2]))
                            })

# Convert the list of small data.frames into a tidy single data.frame
output_df <- dplyr::bind_rows(list_of_dfs)
display(output_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apache Arrow
# MAGIC 
# MAGIC [Apache Arrow](https://arrow.apache.org/) is a project that aims to improve analytics processing performance by representing data in-memory in columnar format and taking advantage of modern hardware.  The main purpose and benefit of the project can be summed up in the following image, taken from the homepage of the project.
# MAGIC 
# MAGIC <img src="https://github.com/kurlare/misc-assets/blob/master/apache-arrow.png?raw=true">
# MAGIC 
# MAGIC Arrow is highly effective at speeding up data transfers.  It's worth mentioning that [Databricks Runtime offers a similar optimization](https://databricks.com/blog/2018/08/15/100x-faster-bridge-between-spark-and-r-with-user-defined-functions-on-databricks.html) out of the box with SparkR.  See [here](https://github.com/marygracemoesta/R-User-Guide/blob/master/Developing_on_Databricks/Customizing.md#apache-arrow-installation) for instructions on how to enable Arrow with `sparklyr`.
# MAGIC 
# MAGIC ___

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Resources
# MAGIC 
# MAGIC This concludes the lesson on UDFs with Spark in R.  If you want to learn more, here are additional resources about distributed R with Spark.
# MAGIC 
# MAGIC 1. [100x Faster Bridge Between R and Spark on Databricks](https://databricks.com/blog/2018/08/15/100x-faster-bridge-between-spark-and-r-with-user-defined-functions-on-databricks.html)
# MAGIC 2. [Shell Oil: Parallelizing Large Simulations using SparkR on Databricks](https://databricks.com/blog/2017/06/23/parallelizing-large-simulations-apache-sparkr-databricks.html)
# MAGIC 3. [Distributed R Chapter from 'The R in Spark'](https://therinspark.com/distributed.html)
# MAGIC 
# MAGIC *** 
