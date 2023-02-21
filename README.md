# TS-Benchmark: An automated benchmarking tool for time series forecasting techniques



This File explains how to set up, extend and operate the Benchmarking Framework.


## Manual Configurations
In this section, the dependencies of the project and manual configurations are described

### java dependencies
The project uses the [Apache Common Maths](https://commons.apache.org/proper/commons-math/) and [JFree Chart](https://www.jfree.org/jfreechart/) packages.
the pom.xml file contains the dependencie for the first package. 
The .jar files of both packages can also be found in the JAR folder.

### python source files
The source files of the used python scripts are situated in the Scripts/Python directory.
The benchmark uses several forecasting models provided by the [DARTS](https://unit8co.github.io/darts/) API. The python API is called from the command line within the java benchmarking source code.
For this, the python interpreter path has to be specified.
This can be done using the set_interpreterpaths() method, further specified in the section "How to setup and customize the benchmarking environment".

The following packages need to be installed to the enviorment of the  python interpreter:

* numpy
* darts
* pandas
* sklearn

### R source files

the benchmark also requires the user to specify the path to a Rscript excutionable using the set_interpreterpaths() method.
The prophet packages has to be installed to the R enviorment.
The source files for the R scripts are provided in the directory Scripts/R



### C source files

TRMF and OTS are implemented in C and the models are again executed over the command line. 
The .exe files (compiled to windows) and the source files are provided in the Scripts/C directory 

## How to setup and customize the benchmarking environment

The src\main\java\Aplication class defines the framework for customizing, executing, and viewing the Benchmark results.
The Class follows the singleton design pattern and therefore the getInstance() method has to be called to create an instance of the benchmark.
Helper methods are provided to customize the methods and datasets used for testing.

When creating the Aplication object for the first time, the user is required to secify the python interpreter-path and the Rscript-path using the set_interpreterpath() method.

To add a new time series to the testing environment the add_time_series() function has to be called.
The add_subseries() command allows the inclusion of a 1D-subseries of a previously added time series to the testing environment.

To add a new method the add_method() function is provided.

With the add_hyperparameter() command the hyperparameter settings are specified. It is possible to set the hyperparameters for all datasets, a time series, and a sub-series individually.
If a hyperparameter is defined in multiple ways the most restrictive hyperparameter setting is used.
In other words, sub-series settings override time series settings which override global settings.

the store() function saves all the performed changes to the setting.txt file.


## How to execute a testing scenario

To test the datasets, the test_datasets() command of the Aplication object has to be executed. As a parameter, the function takes a Test object, which specifies the desired test to be performed.
Three test options have been already implemented:

Standard_Test: evaluates all methods using an 80-20 train-test-split.
Vary_Length: evaluates the forecasting error repeatedly while increasing the size of the time series.
Vary_Dimension: evaluates the forecasting error repeatedly while increasing the dimension of the time series.  

The results of the experiments are written to the Outputs directory and are stored as TSModel.Output and TSModel.Vary_Result objects.

### Adding custom Tests
To add custom tests to the framework a new class has to be created which extends the Aplication.Test class.
This requires the implementation of the test() and file_ending() functions.
the test() function specifies the procedure performed for each time series and method.
the file_ending() function is used to uniquely identify the results file generated by the test.

## How to view the Results of the Experiments

For each Test class, a corresponding Outputter class has been provided which helps navigate and display the results of the experiment.
A stabdard use case of the Aplication object is demonstarted in the main function of the src\main\java\Examples class.



## How to add custom forecasting techniques to the programming environment

The benchmark requires forecasting techniques to extend the src\main\java\thesis\TSModel class. The Benchmark then creates and calls the method using reflection.


### The evaluate() function
The TSModel class requires the implementation of the evaluate() function.
When this function is called the forecasting method is expected to produce its forecast for a provided test and training set.
The test and training set is stored in a private Output object called o.
The training set is stored in the private field Ouput.train of the Output object.
The test set is stored in the private field Output.test.
The forecasting method is expected to set the private field Output.forecast of o to its produced forecast and to set o to the return value of evaluate. The forecast is required to have the same dimensions as the test set.
All three Objects are represented by Matrices, where the row dimension refers to the dimension of the time series and the column dimension to its length.

### The parse() function  

The parse() function is used to provide the method with its hyperparameter settings. the method is passed a Hashmap object, which maps the specified parameter names to their values.
The method has to parse this object and initialize its hyperparameter settings. Refer to one of the provided Method classes for an example of the implementation of this method.

### Dimension-checking functions

is1D(): return true if the method can be applied to one-dimensional time series.

isND(): return true if the method can be applied to multi-dimensional time series.

missingValuesAllowed(): return true if the time series is allowed to contain missing entries.

legalHyperparameters(Matrix training_set): return true when the model can be trained on the time series with the specified hyperparameters. 
					   This function is for example used for the Linear auto-regressive model, to prevent the framework from testing them when their autoregressive order is larger than the training set.




