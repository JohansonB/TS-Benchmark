package thesis.Models;

import thesis.Tools.TimeSeries;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class BATS extends DartsModel {
    public BATS(){
        super();
    }
    public BATS(HashMap<String,ArrayList<String>> in){
        super(in);
    }
    @Override
    protected boolean is_nn() {
        return false;
    }

    @Override
    protected void parse(HashMap<String, ArrayList<String>> in) {
        update(in);
        additionalArguments = "";
        additionalArguments+= in.containsKey("-periods") ? " --periods_TBATS "+in.get("-periods").get(0) : "";
        additionalArguments+= in.containsKey("-trend") ? " --trend_TBATS "+(new Boolean(in.get("-trend").get(0)).booleanValue() ? "True" : "False") : "";

    }
    public static void main(String[] args) throws IOException {
        Vary_Result.load("Outputs\\electricity_consumption\\147\\thesis.Models.GBM_vary_length.txt").plot_at(5);

        Vary_Result.load("Outputs\\electricity_consumption\\147\\thesis.Models.GBM_vary_length.txt").plot_at(6);
    }

    @Override
    public String toString() {
        return "BATS";
    }
}
