package thesis.Models;


import java.util.ArrayList;
import java.util.HashMap;

public class BlockRNN extends DartsModel{
    public BlockRNN(HashMap<String, ArrayList<String>> input) {
        super(input);
    }

    public BlockRNN() {

    }

    @Override
    protected boolean is_nn() {
        return true;
    }

    @Override
    protected void parse(HashMap<String, ArrayList<String>> in) {
        update(in);
        additionalArguments = "";
        additionalArguments +=  in.containsKey("-hidden_size") ? " --hidden_dim "+new Double(in.get("-hidden_size").get(0)).intValue() : "";
        additionalArguments +=  in.containsKey("-n_rnn_layers") ? " --n_rnn_layers "+new Double(in.get("-n_rnn_layers").get(0)).intValue() : "";


    }
    public String toString(){
        return "BlockRNN";
    }


}
