import thesis.Aplication;
import thesis.Models.SeasonalRandomWalk;
import thesis.Tools.Metric;

import java.util.HashMap;

public class Examples {
    //this illustrates how to operate the benchmark using the thesis.Aplication object
    public static void main(String[] args){
        //create thesis.Aplication
        Aplication a = Aplication.getInstance();
        //add new Method to the benchmark
        a.add_method(new SeasonalRandomWalk());
        //add hyperparameter settings for all datasets
        HashMap<String, String> parameter_mapping = new HashMap<>();
        parameter_mapping.put("-period","1");
        a.add_hyperparameters(new SeasonalRandomWalk(),parameter_mapping);

        //add hyperparameter settings for individual time series
        parameter_mapping = new HashMap<>();
        parameter_mapping.put("-period","7");
        a.add_hyperparameters(new SeasonalRandomWalk(),"ATM_withdraw",parameter_mapping);

        //store the setting to the app commented out since I dont want to apply these changes to the settings file
        //a.save();

        //perform a test on the datasets and methods
        a.test_datasets(new Aplication.Standard_Test());

        //plot the results of the Standard test with the specified metric
        Aplication.Vary_Horizon v = new Aplication.Vary_Horizon(new Metric.RMSE());
        v.run();
        v.plot_graphs();

    }
}
