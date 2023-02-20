package thesis;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import thesis.Models.*;
import thesis.Tools.*;

import java.io.*;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.*;
import java.util.stream.Collectors;

public abstract class TSModel implements GridSearcher.Gridsearchable{
    //public static boolean normalization = true;
    protected Integer stepSize = null;
    protected boolean tune_hyperParams = false;
    private boolean stepSize_manually_set = false;


    protected Output o;
    protected RealMatrix ts;
    protected double split;
    public TSModel(HashMap<String, ArrayList<String>> in){
        if(in.containsKey("-predictionHorizon")){
            stepSize = new Integer(in.get("-predictionHorizon").get(0));
        }
        parse(in);
    }
    public TSModel(){
        parse(new HashMap<>());
    }

    public static TSModel String_to_model(String model, HashMap<String,String> in) {
        if(model.equalsIgnoreCase("AutoARIMA")){
            return new ARIMA(GridSearcher.reformat(in));
        }
        else if(model.equalsIgnoreCase("beatlex")){
            return new Beatlex(GridSearcher.reformat(in));
        }
        else if(model.equalsIgnoreCase("BlockRNN")){
            return new BlockRNN(GridSearcher.reformat(in));

        }
        else if(model.equalsIgnoreCase("FFT")){
            return new FFT(GridSearcher.reformat(in));
        }
        else if(model.equalsIgnoreCase("LSRN")){
            return new LSRN(GridSearcher.reformat(in));
        }
        else if(model.equalsIgnoreCase("MESVD")){
            return new ME(GridSearcher.reformat(in));
        }
        else if(model.equalsIgnoreCase("NBEATS")){
            return new N_BEATS(GridSearcher.reformat(in));

        }
        else if(model.equalsIgnoreCase("ots")){
            return new OTS(GridSearcher.reformat(in));
        }
        else if(model.equalsIgnoreCase("prophet")){
            return new Prophet(GridSearcher.reformat(in));
        }
        else if(model.equalsIgnoreCase("ETS")){
            return new SES(GridSearcher.reformat(in));
        }
        else if(model.equalsIgnoreCase("TCNM")){
            return new TCNModel(GridSearcher.reformat(in));
        }
        else if(model.equalsIgnoreCase("TFT")){
            return new TFTModel(GridSearcher.reformat(in));
        }
        else if(model.equalsIgnoreCase("Theta")){
            return new ThetaForecaster(GridSearcher.reformat(in));
        }
        else if(model.equalsIgnoreCase("TransformerModel")){
            return new TransformerModel(GridSearcher.reformat(in));

        }
        else if(model.equalsIgnoreCase("trmf")){
            return new trmf(GridSearcher.reformat(in));
        }
        else if(model.equalsIgnoreCase("TBATS")){
            return new TBATS(GridSearcher.reformat(in));
        }
        else if(model.equalsIgnoreCase("BATS")){
            return new BATS(GridSearcher.reformat(in));
        }
        else if(model.equalsIgnoreCase("RandomForest")){
            return new RandomForest(GridSearcher.reformat(in));
        }
        else if(model.equalsIgnoreCase("GBM")){
            return new GBM(GridSearcher.reformat(in));
        }
        else {
            throw new Error("method does not exist");
        }
    }

    protected abstract void parse(HashMap<String, ArrayList<String>> in);

    public void update_params(HashMap<String, ArrayList<String>> search_crossValid) {
        parse(search_crossValid);
    }

    public static TSModel dummy() {
        return new ARIMA();
    }


    public static class Vary_Result{
        HashMap<Double,Output> experiment_outputs;
        String data_path;
        HashMap<String,String> params;
        String model;
        Mode mode;

        public Output Output_at(Double key) {
            return experiment_outputs.get(key);
        }

        public HashMap<Double,Output> get_results() {
            return experiment_outputs;
        }

        public Output tail() {
            Double max = Collections.max(experiment_outputs.keySet());
            return experiment_outputs.get(max);
        }

        public Graph error_graph(Metric m,String data_name, String method){
            Graph g = new Graph();
            g.dots(true);
            g.set_titel("Error Graph");
            g.set_value_string(m.getDescription());
            g.set_data_set(data_name);
            g.set_indexs(new IndexSet(mode==Mode.VARY_DIMENSION ? "number dimensions" : "number data points"));
            //experiment_outputs.forEach((k,v)->g.add(v.getModel(),mode.step_size(v,k),v.error(m)));
            ArrayList<Double> keys = new ArrayList<>();
            keys.addAll(experiment_outputs.keySet());
            Collections.sort(keys);
            keys.forEach(v -> {
                g.add(method,v,experiment_outputs.get(v).error(m));
            });

            return g;
        }

        public Graph runtime_graph(String data_name,String method){
            Graph g = new Graph();
            g.dots(true);
            g.set_value_string("Runtime in seconds");
            g.set_titel("Runtime plot");
            g.set_data_set(data_name);
            //g.set_log_scale_y(true);
            g.set_log_base_y(10);
            g.set_indexs(new IndexSet(mode.description()));
            ArrayList<Double> keys = new ArrayList<>();
            keys.addAll(experiment_outputs.keySet());
            Collections.sort(keys);
            keys.forEach(v -> {
                g.add(method,v,experiment_outputs.get(v).getRunTime());
            });

            return g;
        }

        public enum Mode{VARY_DIMENSION, VARY_LENGTH,VARY_PARAMETER;

            public String description() {
                if (this == VARY_DIMENSION)
                    return "number time series";
                else if (this == VARY_LENGTH)
                    return "number data points";
                else if(this == VARY_PARAMETER)
                    return "parameter Space";
                else{
                    throw new Error("behavior not defined for this Mode");
                }

            }

            public double step_size(Output v, double k) {
                if (this == VARY_DIMENSION)
                    return v.getTrain().getRowDimension();
                else if (this == VARY_LENGTH)
                    return v.getTrain().getColumnDimension();
                else if(this == VARY_PARAMETER)
                    return k;
                else{
                    throw new Error("behavior not defined for this Mode");
                }

            }
        }
        public Vary_Result(String data_path,HashMap<Double,Output> experiment_outputs, HashMap<String,String> params,Mode mode){
            this.experiment_outputs = experiment_outputs;
            this.data_path = data_path;
            this.params = params;
            model = experiment_outputs.values().iterator().next().getModel();
            this.mode = mode;


        }
        public Vary_Result(){
        }
        public Vary_Result subseries(int index) {
            Vary_Result ret = new Vary_Result();
            ret.experiment_outputs = new HashMap<>();
            ret.data_path = data_path;
            ret.model = model;
            ret.mode = mode;
            experiment_outputs.forEach((k,v)->
                ret.experiment_outputs.put(k,v.sub_copy(index))
            );
            return ret;
        }
        public String encode(){
            StringBuilder sb = new StringBuilder();
            sb.append(mode.toString());
            sb.append("###");
            sb.append(model);
            sb.append("###");
            sb.append(encode_Hashmap(params));
            sb.append("###");
            sb.append(data_path);
            sb.append("###");
            sb.append(encode_Hashmap(experiment_outputs));
            return sb.toString();

        }
        public void store(String path) throws IOException {
            File f = new File(path);
            f.getParentFile().mkdirs();
            BufferedWriter writer = new BufferedWriter(new FileWriter(f));
            writer.write(mode.toString());
            writer.write("###");
            writer.write(model);
            writer.write("###");
            writer.write(encode_Hashmap(params));
            writer.write("###");
            writer.write(data_path);
            writer.write("###");
            writer.write(encode_Hashmap(experiment_outputs));
            writer.flush();






        }
        public Graph runtime_graph(){
            Graph g = new Graph();
            g.dots(true);
            g.set_value_string("Runtime in seconds");
            g.set_titel("Runtime plot");
            g.set_data_set(data_path);
            //g.set_log_scale_y(true);
            g.set_log_base_y(10);
            g.set_indexs(new IndexSet(mode.description()));
            ArrayList<Double> keys = new ArrayList<>();
            keys.addAll(experiment_outputs.keySet());
            Collections.sort(keys);
            keys.forEach(v -> {
                g.add(experiment_outputs.get(v).getModel(),v,experiment_outputs.get(v).getRunTime());
            });

            return g;
        }
        public Graph runtime_graph(String data_name){
            Graph g = new Graph();
            g.dots(true);
            g.set_value_string("Runtime in seconds");
            g.set_titel("Runtime plot");
            g.set_data_set(data_name);
            //g.set_log_scale_y(true);
            g.set_log_base_y(10);
            g.set_indexs(new IndexSet(mode.description()));
            ArrayList<Double> keys = new ArrayList<>();
            keys.addAll(experiment_outputs.keySet());
            Collections.sort(keys);
            keys.forEach(v -> {
                g.add(experiment_outputs.get(v).getModel(),v,experiment_outputs.get(v).getRunTime());
            });

            return g;
        }
        public Graph error_graph(Metric m){
            Graph g = new Graph();
            g.dots(true);
            g.set_titel("Error Graph");
            g.set_value_string(m.getDescription());
            g.set_data_set(data_path);
            g.set_indexs(new IndexSet(mode==Mode.VARY_DIMENSION ? "number dimensions" : "number data points"));
            //experiment_outputs.forEach((k,v)->g.add(v.getModel(),mode.step_size(v,k),v.error(m)));
            ArrayList<Double> keys = new ArrayList<>();
            keys.addAll(experiment_outputs.keySet());
            Collections.sort(keys);
            keys.forEach(v -> {
                g.add(experiment_outputs.get(v).getModel(),v,experiment_outputs.get(v).error(m));
            });

            return g;
        }
        public Graph error_graph(Metric m,String data_name){
            Graph g = new Graph();
            g.dots(true);
            g.set_titel("Error Graph");
            g.set_value_string(m.getDescription());
            g.set_data_set(data_name);
            g.set_indexs(new IndexSet(mode==Mode.VARY_DIMENSION ? "number dimensions" : "number data points"));
            //experiment_outputs.forEach((k,v)->g.add(v.getModel(),mode.step_size(v,k),v.error(m)));
            ArrayList<Double> keys = new ArrayList<>();
            keys.addAll(experiment_outputs.keySet());
            Collections.sort(keys);
            keys.forEach(v -> {
                g.add(experiment_outputs.get(v).getModel(),v,experiment_outputs.get(v).error(m));
            });

            return g;
        }
        public void plot_at(int rank,int cap){
            List<Double> temp = new ArrayList<>(experiment_outputs.keySet());
            Collections.sort(temp);
            experiment_outputs.get(temp.get(rank-1)).plotPairwiseComparrison(cap);

        }
        public Output Output_at(int rank){
            List<Double> temp = new ArrayList<>(experiment_outputs.keySet());
            Collections.sort(temp);
            return experiment_outputs.get(temp.get(rank-1));

        }
        public void plot_at(int rank){
            List<Double> temp = new ArrayList<>(experiment_outputs.keySet());
            Collections.sort(temp);
            experiment_outputs.get(temp.get(rank-1)).plotComparrison();

        }
        public void plot_at(double key){
            experiment_outputs.get(key).plotPairwiseComparrison();

        }
        public static Rank rank(String directory,int index, Metric m) throws IOException {
            File dir = new File(directory);
            Rank r = new Rank();
            HashMap<String,Double> erorrs = new HashMap<>();
            for(File f : dir.listFiles()){
                System.out.println(f.getName());
                if(f.isDirectory()){
                    if(f.getName().equalsIgnoreCase(new Integer(index).toString())){
                        for(File fi : f.listFiles()){
                            System.out.println(fi.getName());
                            if(fi.getName().contains("v_d")){
                                Vary_Result v_d_error = load(fi.getPath());
                                Double key = Collections.max(v_d_error.experiment_outputs.keySet());
                                Output o = v_d_error.experiment_outputs.get(key);

                                //marker to delete
                                if(index == 246) {
                                    o.setTest(o.getTest().getSubMatrix(0, 0, 0, o.getTest().getColumnDimension() - 121));
                                    o.setForecast(o.getForecast().getSubMatrix(0, 0, 0, o.getForecast().getColumnDimension() - 121));
                                }


                                erorrs.put(o.getModel(),o.error(m));
                            }

                        }
                    }

                }
                if(f.getName().contains("v_d")){
                    Vary_Result v_d_error = load(f.getPath());
                    Double key = Collections.max(v_d_error.experiment_outputs.keySet());
                    Output o = v_d_error.experiment_outputs.get(key);

                    RealMatrix test = o.test;
                    RealMatrix forecast = o.forecast;
                    RealMatrix train = o.train;
                    o.test = o.test.getRowMatrix(index);
                    o.forecast = o.forecast.getRowMatrix(index);
                    o.train = o.train.getRowMatrix(index);

                    //marker to delete
                    if(index == 246) {
                        o.setTest(o.getTest().getSubMatrix(0, 0, 0, o.getTest().getColumnDimension() - 121));
                        o.setForecast(o.getForecast().getSubMatrix(0, 0, 0, o.getForecast().getColumnDimension() - 121));
                    }


                    erorrs.put(o.getModel(),o.error(m));
                    o.test = test;
                    o.forecast = forecast;
                    o.train = train;


                }
            }
            ArrayList<Double> erorrs_l = new ArrayList<>(erorrs.values());
            Collections.sort(erorrs_l);

            for(Double d : erorrs_l){
                String key = erorrs.entrySet().stream()
                        .filter(entry -> d.equals(entry.getValue()))
                        .findFirst().map(Map.Entry::getKey)
                        .orElse(null);
                erorrs.remove(key);
                System.out.println(key);
                r.rank_list.add(key);
                r.error.add(d);



            }

            return r;

        }
        public static Graph all_horizon(String directory,Metric m,int step_size, boolean truncate) throws IOException {
            File dir = new File(directory);
            Graph g = null;
            for(File f : dir.listFiles()){
                System.out.println(f.getName());
                if(f.isDirectory()){
                    continue;
                }
                if(f.getName().contains("v_d")){
                    Vary_Result v_d_error = load(f.getPath());
                    Double key = Collections.max(v_d_error.experiment_outputs.keySet());
                    Output o = v_d_error.experiment_outputs.get(key);
                    if(g == null) {
                        g=o.all_horizon_graph(m,step_size,truncate);
                    }
                    else{
                        g=g.merge(o.all_horizon_graph(m,step_size,truncate));
                    }


                }

            }
            return g;
        }
        public static Graph all_horizon(String directory,int index,Metric m, int step_size,boolean truncate) throws IOException {
            File dir = new File(directory);
            Graph g = null;
            for(File f : dir.listFiles()){
                System.out.println(f.getName());
                if(f.isDirectory()){
                    if(f.getName().equalsIgnoreCase(new Integer(index).toString())){
                        for(File fi : f.listFiles()){
                            System.out.println(fi.getName());
                            if(fi.getName().contains("v_d")){
                                Vary_Result v_d_error = load(fi.getPath());
                                Double key = Collections.max(v_d_error.experiment_outputs.keySet());
                                Output o = v_d_error.experiment_outputs.get(key);

                                if(g == null) {
                                    g=o.all_horizon_graph(m,step_size,truncate);
                                }
                                else {
                                    g = g.merge(o.all_horizon_graph(m,step_size,truncate));
                                }
                            }

                        }
                    }

                }
                if(f.getName().contains("v_d")){
                    Vary_Result v_d_error = load(f.getPath());
                    Double key = Collections.max(v_d_error.experiment_outputs.keySet());
                    Output o = v_d_error.experiment_outputs.get(key);

                    RealMatrix test = o.test;
                    RealMatrix forecast = o.forecast;
                    RealMatrix train = o.train;
                    o.test = o.test.getRowMatrix(index);
                    o.forecast = o.forecast.getRowMatrix(index);
                    o.train = o.train.getRowMatrix(index);
                    if(g == null) {
                        g=o.all_horizon_graph(m,step_size,truncate);
                    }
                    else{
                        g=g.merge(o.all_horizon_graph(m,step_size,truncate));
                    }
                    o.test = test;
                    o.forecast = forecast;
                    o.train = train;


                }

            }
            return g;
        }



        public static Vary_Result load(String path) throws IOException {
            Vary_Result v = new Vary_Result();
            File f = new File(path);
            BufferedReader reader = new BufferedReader(new FileReader(f));
            String encoded = reader.lines().collect(Collectors.joining(System.lineSeparator()));
            String[] components = encoded.split("###");
            v.mode = Mode.valueOf(components[0]);
            TSModel model = TSModel.String_to_model(components[1],decode_modelparams(components[2]));
            v.params = decode_modelparams(components[2]);
            v.data_path = components[3];
            v.experiment_outputs = decode_experiment_outputs(components[4],model);
            v.model = model.toString();
            return v;
        }
        public static Vary_Result decode_v_r(String encoding) throws IOException {
            Vary_Result v = new Vary_Result();
            String[] components = encoding.split("###");
            v.mode = Mode.valueOf(components[0]);
            TSModel model = TSModel.String_to_model(components[1],decode_modelparams(components[2]));
            v.params = decode_modelparams(components[2]);
            v.data_path = components[3];
            v.experiment_outputs = decode_experiment_outputs(components[4],model);
            v.model = model.toString();
            return v;
        }

        private static HashMap<Double, Output> decode_experiment_outputs(String component,TSModel model) {
            HashMap<Double,Output> ret = new HashMap<>();
            String[] entries = component.split("\n");
            for(String s : entries){
                if(s.length()!= 0){
                    String[] pair = s.split("##");
                    ret.put(new Double(pair[0]),decode_output(pair[1],model));
                }
            }
            return ret;

        }



        private static HashMap<String, String> decode_modelparams(String component) {
            HashMap<String,String> ret = new HashMap<>();
            String[] entries = component.split("\n");
            for(String s : entries){
                if(s.length()!= 0){
                    String[] pair = s.split("##");
                    ret.put(pair[0],pair[1]);
                }
            }
            return ret;
        }

        private String encode_Hashmap(HashMap h){
            StringBuilder sb = new StringBuilder();
            h.forEach((k,v)-> {
                sb.append(k.toString());
                sb.append("##");
                sb.append(v.toString());
                sb.append("\n");
            }
            );
            return sb.toString();
        }

        public static class Rank {
            public ArrayList<String> rank_list = new ArrayList<>();
            public ArrayList<Double> error = new ArrayList<>();

            public void print(){
                for(int i = 0; i<rank_list.size();i++){
                    System.out.println( (i+1)+". "+rank_list.get(i)+" error: "+error.get(i));
                }
            }
            public static HashMap<String,Double> score(String path, Metric m, int ... indxs) throws IOException {
                HashMap<String,Double> score = new HashMap<>();
                if(indxs.length==0) {
                    HashMap<String,Double> errors = flatten(Vary_Result.all_horizon(path, m,10000,true).get_ys());
                    errors.forEach((k,v) -> score.put(k,v));
                }
                else{
                    HashMap<String,ArrayList<Double>> scores = new HashMap<>();
                    for(int i = 0; i<indxs.length;i++){
                        HashMap<String,Double> errors = flatten(Vary_Result.all_horizon(path,indxs[0], m,10000,true).get_ys());
                        Double max = Collections.max(errors.values());
                        if(i == 0){
                            errors.forEach((k,v)-> scores.put(k,new ArrayList<>()));
                        }
                        errors.forEach((k,v)->scores.get(k).add(v));

                    }
                    scores.forEach((k,v)-> score.put(k,new Mean().evaluate(ArrayUtils.toPrimitive(scores.get(k).toArray(new Double[scores.get(k).size()])))));
                }
                return score;
            }
            public static HashMap<String,Integer> nominal(String path, Metric m, int ... indxs) throws IOException {
                HashMap<String,Integer> nominal = new HashMap<>();
                HashMap<String,Double> score = score(path, m, indxs);
                score.forEach((k,v) -> nominal.put(k,nominal_linear(v)));
                return nominal;
            }

            private static HashMap<String, Double> flatten(HashMap<String, ArrayList<Double>> ys) {
                HashMap<String,Double> ret = new HashMap<>();
                ys.forEach((k,v)-> ret.put(k,ys.get(k).get(0)));
                return ret;
            }

            private static Integer nominal_linear(Double v) {
                return 4  - (v >= 0.5 ? 4 : (int)(v/0.1));

            }


        }
    }

    public Output test_accuracy(RealMatrix train, RealMatrix test){

        this.ts = new TimeSeries(train).append(new TimeSeries(test)).toMatrix();
        this.split = 1-((double)test.getColumnDimension())/train.getColumnDimension();
        o = new Output();
        o.train = train;
        o.test = test;

        //check if data has correct format, if not return the empty output to signalize that the test could not be performed
        if((train.getRowDimension()==1 && !is1D()) || (train.getRowDimension()>1 && !isND()) || !legal_hyperparameters(train)){
            o.invalidate();
            return o;
        }

        return evaluate();

    }

    public Output test_accuracy(RealMatrix ts, double split){

        this.ts = ts;
        this.split = split;
        o = new Output(ts, split);
        update_stepSize(o.getTest().getColumnDimension());

        //check if data has correct format, if not return the empty output to signalize that the test could not be performed
        if((o.train.getRowDimension()==1 && !is1D()) || (o.train.getRowDimension()>1 && !isND()) || !legal_hyperparameters(o.train)){
            o.invalidate();
            return o;
        }

        return evaluate();

    }

    protected abstract Output evaluate();

    public Output test_accuracy(String ts, double split) throws IOException {
        return test_accuracy(new TimeSeries(ts).toMatrix(),split);

    }
    public Output[] test_runtime(int init_n, int init_m, double init_split, int n_step, int m_step, double split_step, int nbStep){
        Output[] ret = new Output[nbStep];
        for(int i = 0;  i<nbStep;i++){
            ret[i] = test_accuracy(MatrixTools.randomMatrix(init_n+i*n_step,init_m+i*m_step,-1,1),init_split-i*split_step);
        }
        return ret;
    }
    protected void update_stepSize(int len){
        stepSize = !stepSize_manually_set ? len : stepSize;
    }

    public void setStepSize(int stepSize){
        this.stepSize = stepSize;
        stepSize_manually_set = true;
    }

    public Vary_Result vary_length(String data_name, TimeSeries ts, double initial_split, double step_size) throws IOException {
        String local_data_path = "Datasets\\Temp\\vary_length_temp.csv";
        double predictionRate = 0.2*initial_split;
        RealMatrix dat = ts.toMatrix();
        int absolutLength = (int)(dat.getColumnDimension()*predictionRate);
        HashMap<Double,Output> h = new HashMap<>();
        for(double j = initial_split;round(j,3)<=1;j+=step_size){
            RealMatrix cur = dat.getSubMatrix(0,dat.getRowDimension()-1,(int)((1-j)*(dat.getColumnDimension()-1)),dat.getColumnDimension()-1);
            new TimeSeries(cur).writeToCSV(local_data_path);
            double curSplit = 1-(double)absolutLength/cur.getColumnDimension();
            h.put(round(j,3),test_accuracy(local_data_path,curSplit));
        }
        return new Vary_Result(data_name,h,params(), Vary_Result.Mode.VARY_LENGTH);
    }
    public Vary_Result vary_length_percent(String data_name, TimeSeries ts, double initial_split, int steps) throws IOException {
        String local_data_path = "Datasets\\Temp\\vary_length_temp.csv";
        double predictionRate = 0.2*initial_split;
        RealMatrix dat = ts.toMatrix();
        int absolutLength = (int)(dat.getColumnDimension()*predictionRate);
        double percent = Math.pow(1./initial_split,1./steps);
        double init_length = dat.getColumnDimension()*initial_split;
        HashMap<Double,Output> h = new HashMap<>();
        for(int j = 0;j<=steps;j++){
            int length = (int)Math.ceil(init_length*Math.pow(1+percent,j));
            RealMatrix cur = dat.getSubMatrix(0,dat.getRowDimension()-1,dat.getColumnDimension()-length,dat.getColumnDimension()-1);
            new TimeSeries(cur).writeToCSV(local_data_path);
            double curSplit = 1-(double)absolutLength/cur.getColumnDimension();
            h.put(new Double(length),test_accuracy(local_data_path,curSplit));
        }
        return new Vary_Result(data_name,h,params(), Vary_Result.Mode.VARY_LENGTH);
    }

    public HashMap<String, String> params() {
        return new HashMap<>();
    }

    public Vary_Result vary_length(String data_path) throws IOException {
            return vary_length(data_path,new TimeSeries(data_path), 0.4, 0.1);

    }

    public Vary_Result vary_length(TimeSeries ts,String data_name) throws IOException {
        return vary_length(data_name,ts, 0.4, 0.1);

    }
    public Vary_Result vary_dimension(String data_name, TimeSeries ts, int init_dim, int step_size, double split) throws IOException {
        if (step_size==0){
            step_size=1;
        }
        String local_data_path = "Datasets\\Temp\\vary_dimension_temp.csv";
        RealMatrix dati = ts.toMatrix();
        HashMap<Double,Output> h = new HashMap<>();
        int j = init_dim;
        while (j<=dati.getRowDimension()){
            System.out.println(j);
            new TimeSeries(dati.getSubMatrix(0, j - 1, 0, dati.getColumnDimension() - 1)).writeToCSV(local_data_path);
            h.put(new Double(j),test_accuracy(local_data_path,split));

            if(j==dati.getRowDimension()){
                break;
            }
            j += step_size;
            if(j>=dati.getRowDimension()){
                j = dati.getRowDimension();
            }
        }
        return new Vary_Result(data_name,h,params(), Vary_Result.Mode.VARY_DIMENSION);

    }
    public Vary_Result vary_dimension(TimeSeries ts, String data_name) throws IOException{
        return vary_dimension(data_name,ts,1,(int)(ts.toMatrix().getRowDimension()/10), 0.8);
    }
    public Vary_Result vary_dimension(String data_path) throws IOException {
        return vary_dimension(data_path,new TimeSeries(data_path),1,(int)(new TimeSeries(data_path).toMatrix().getRowDimension()/10), 0.8);
    }
    public Vary_Result vary_parameter(String data_path, double lower_bound, double upper_bound,double step_size, String parameter, HashMap<String,String> Model_configuration) throws IOException {
        HashMap<Double, Output> h = new HashMap<>();
        HashMap<String,String> model = (HashMap<String, String>) Model_configuration.clone();
        for(double i = lower_bound;i<=upper_bound;i += step_size) {
            System.out.println(i);
            //update Model Hperparmaters
            model.put(parameter,new Double(i).toString());
            parse(GridSearcher.reformat(model));

            h.put(new Double(i), test_accuracy(data_path, 0.8));

        }
        return new Vary_Result(data_path,h,params(), Vary_Result.Mode.VARY_PARAMETER);

    }
    public Vary_Result vary_parameter(String data_path, ArrayList<Double> steps, String parameter, HashMap<String,String> Model_configuration) throws IOException {
        HashMap<Double, Output> h = new HashMap<>();
        HashMap<String,String> model = (HashMap<String, String>) Model_configuration.clone();
        for(Double i : steps) {
            System.out.println(i);
            //update Model Hperparmaters
            model.put(parameter,i.toString());
            parse(GridSearcher.reformat(model));

            h.put(i, test_accuracy(data_path, 0.8));

        }
        return new Vary_Result(data_path,h,params(), Vary_Result.Mode.VARY_PARAMETER);

    }

    public static ArrayList<Double> additive_range(double lower_bound, double upper_bound,double step_size){
        ArrayList<Double> ret = new ArrayList<>();
        for(double i = lower_bound;i<=upper_bound;i += step_size) {
            ret.add(i);
        }
        return ret;

    }
    public static ArrayList<Double> mult_range(double lower_bound, double upper_bound,double step_size){
        ArrayList<Double> ret = new ArrayList<>();
        for(double i = lower_bound;i<=upper_bound;i *= step_size) {
            ret.add(i);
        }
        return ret;

    }
    public static ArrayList<Double> exponent_range(int base, int low, int highe){
        ArrayList<Double> ret = new ArrayList<>();
        for(int i = low;i<=highe;i++){
            ret.add(Math.pow(base,i));
        }
        return ret;
    }

    public abstract boolean is1D();
    public abstract boolean isND();
    public abstract boolean legal_hyperparameters(RealMatrix train);
    public abstract boolean missingValuesAllowed();




    private static double round (double value, int precision) {
        int scale = (int) Math.pow(10, precision);
        return (double) Math.round(value * scale) / scale;
    }














    public class Output{
        private double runTime;
        private RealMatrix forecast;
        private RealMatrix train;
        private RealMatrix test;
        private RealMatrix ts;
        private double split;
        public Output(RealMatrix ts, double split){
            this.ts = ts;
            this.split = split;
            Tuple<RealMatrix,RealMatrix> t  = MatrixTools.split(ts,split);
            train = t.getVal1();
            test = t.getVal2();

        }
        public Output(){

        }

        public Output(RealMatrix train, RealMatrix test, RealMatrix forecast) {
            this.train = train;
            this.forecast = forecast;
            this.test = test;
        }
        public Output(RealMatrix train, RealMatrix test, RealMatrix forecast,double runTime) {
            this.train = train;
            this.forecast = forecast;
            this.test = test;
            this.runTime = runTime;
        }

        public Graph all_horizon_graph(Metric m){
            return Graph.all_horizon_graph(this,m);
        }
        public Graph all_horizon_graph(Metric m,int step_size,boolean truncate){
            return Graph.all_horizon_graph_aggregated(this,m,step_size,truncate);
        }

        public void invalidate(){
            train = null;
            test = null;
            forecast=null;
            runTime = Double.NaN;
        }
        public boolean is_invalid(){
            if(train==null&&test==null&&forecast==null){
                return true;
            }
            if((test.getColumnDimension()==1&&test.getRowDimension()==1&&Double.isNaN(test.getEntry(0,0)))) {
                return true;
            }

            return false;
        }

        public String toString(){
            StringBuilder ret = new StringBuilder();
            ret.append(runTime);
            ret.append("%%%");
            if(train==null){
                ret.append("!");
            }
            else {
                for (int i = 0; i < train.getRowDimension(); i++) {
                    for (int j = 0; j < train.getColumnDimension(); j++) {
                        ret.append(train.getEntry(i, j));
                        ret.append(" ");
                    }
                    ret.append("%");
                }
            }
            ret.append("%%%");
            if(test==null){
                ret.append("!");
            }
            else {
                for (int i = 0; i < test.getRowDimension(); i++) {
                    for (int j = 0; j < test.getColumnDimension(); j++) {
                        ret.append(test.getEntry(i, j));
                        ret.append(" ");
                    }
                    ret.append("%");
                }
            }
            ret.append("%%%");
            if(forecast==null){
                ret.append("!");
            }
            else {
                for (int i = 0; i < forecast.getRowDimension(); i++) {
                    for (int j = 0; j < forecast.getColumnDimension(); j++) {
                        ret.append(forecast.getEntry(i, j));
                        ret.append(" ");
                    }
                    ret.append("%");
                }
            }
            ret.append("%%%");
            ret.append(TSModel.this.getClass().getName());
            return ret.toString();
        }


        public void plotComparrison(){
            plotComparrison(true,"");
        }
        public Graph plotComparrison(String titel, String dataset){
            RealMatrix tester = MatrixUtils.createRealMatrix(test.getRowDimension(),test.getColumnDimension()+train.getColumnDimension());
            RealMatrix verify = MatrixUtils.createRealMatrix(tester.getRowDimension(),tester.getColumnDimension());


            tester.setSubMatrix(train.getData(),0,0);
            tester.setSubMatrix(test.getData(),0,train.getColumnDimension());
            verify.setSubMatrix(train.getData(),0,0);

            verify.setSubMatrix(MatrixTools.infinityToNan(forecast).getData(),0,train.getColumnDimension());

            Graph g = new Graph(dataset,titel,"timeSteps","value",new IndexSet("time",0,tester.getColumnDimension()));
            g.add("test",tester);
            g.add("validation",verify);

            g.plot();

            return g;
        }
        public Graph plotComparrison(boolean showPlot,String datasetName ){

            RealMatrix tester = MatrixUtils.createRealMatrix(test.getRowDimension(),test.getColumnDimension()+train.getColumnDimension());
            RealMatrix verify = MatrixUtils.createRealMatrix(tester.getRowDimension(),tester.getColumnDimension());


            tester.setSubMatrix(train.getData(),0,0);
            tester.setSubMatrix(test.getData(),0,train.getColumnDimension());
            verify.setSubMatrix(train.getData(),0,0);

            verify.setSubMatrix(MatrixTools.infinityToNan(forecast).getData(),0,train.getColumnDimension());

            Graph g = new Graph(datasetName,TSModel.this.toString(),"timeSteps","value",new IndexSet("time",0,tester.getColumnDimension()));
            g.add("validation",verify);
            g.add("test",tester);
            if(showPlot)
                g.plot();
            return g;
        }
        public String getModel(){
            return TSModel.this.toString();
        }
        public Graph plotToGraph() {
            RealMatrix tester = MatrixUtils.createRealMatrix(test.getRowDimension(),test.getColumnDimension()+train.getColumnDimension());
            RealMatrix verify = MatrixUtils.createRealMatrix(tester.getRowDimension(),tester.getColumnDimension());


            tester.setSubMatrix(train.getData(),0,0);
            tester.setSubMatrix(test.getData(),0,train.getColumnDimension());
            verify.setSubMatrix(train.getData(),0,0);

            verify.setSubMatrix(MatrixTools.infinityToNan(forecast).getData(),0,train.getColumnDimension());

            Graph g = new Graph(new IndexSet("time",0,tester.getColumnDimension()));
            return g;
        }
        public void plotPairwiseComparrison() {
            RealMatrix tester = MatrixUtils.createRealMatrix(test.getRowDimension(),test.getColumnDimension()+train.getColumnDimension());
            RealMatrix verify = MatrixUtils.createRealMatrix(tester.getRowDimension(),tester.getColumnDimension());


            tester.setSubMatrix(train.getData(),0,0);
            tester.setSubMatrix(test.getData(),0,train.getColumnDimension());
            verify.setSubMatrix(train.getData(),0,0);
            verify.setSubMatrix(MatrixTools.infinityToNan(forecast).getData(),0,train.getColumnDimension());

            Graph[] graphList = new Graph[verify.getRowDimension()];
            for(int i = 0; i<verify.getRowDimension();i++){
                graphList[i] = new Graph(new IndexSet("time",0,tester.getColumnDimension()));
                graphList[i].add("validation",verify.getRowMatrix(i));
                graphList[i].add("test",tester.getRowMatrix(i));
                graphList[i].plot();


            }

        }
        public void plot_range(int start, int end, int cap) {
            Output o = new Output();
            o.train = MatrixUtils.createRealMatrix(train.getRowDimension(),1);
            o.forecast = forecast.getSubMatrix(0,forecast.getRowDimension()-1,start,end);
            o.test = test.getSubMatrix(0,train.getRowDimension()-1,start,end);
            o.plotPairwiseComparrison(cap);
        }
        public void block_plot(int step_size, int cap) {
            int len = step_size;
            for( int index = 0;index<forecast.getColumnDimension();index+= step_size ){
                if(forecast.getColumnDimension()-1-index<step_size)
                    len = forecast.getColumnDimension()-1-index;
                else
                    len = step_size;
                plot_range(index,index+len, cap);
            }
        }
        public void block_plot(int step_size){
            block_plot(step_size, 1);
        }
        public void plotPairwiseComparrison(int cap) {
            RealMatrix tester = MatrixUtils.createRealMatrix(test.getRowDimension(), test.getColumnDimension() + train.getColumnDimension());
            RealMatrix verify = MatrixUtils.createRealMatrix(tester.getRowDimension(), tester.getColumnDimension());


            tester.setSubMatrix(train.getData(), 0, 0);
            tester.setSubMatrix(test.getData(), 0, train.getColumnDimension());
            verify.setSubMatrix(train.getData(), 0, 0);
            verify.setSubMatrix(MatrixTools.infinityToNan(forecast).getData(), 0, train.getColumnDimension());

            Graph[] graphList = new Graph[verify.getRowDimension()];
            for (int i = 0; i < verify.getRowDimension() && i < cap; i++) {
                graphList[i] = new Graph(new IndexSet("time", 0, tester.getColumnDimension()));
                graphList[i].add("validation", verify.getRowMatrix(i));
                graphList[i].add("test", tester.getRowMatrix(i));
                graphList[i].plot();


            }
        }


        public RealMatrix getForecast(){
            return forecast;
        }
        public RealMatrix getTrain(){
            return train;
        }
        public RealMatrix getTest(){
            return test;
        }
        public RealMatrix getTs(){
            return ts;
        }

        public double getSplit() {
            return split;
        }
        public double getRunTime(){
            return runTime;
        }

        public void setRunTime(double runTime) {
            this.runTime = runTime;
        }

        public void setForecast(RealMatrix forecast) {
            this.forecast = forecast;
        }

        public void setTest(RealMatrix mat) {
            test = mat;
        }

        public void setTrain(RealMatrix realMatrix) {
            train = realMatrix;
        }

        public double error(Metric m){
            if(is_invalid()){
                return Double.NaN;
            }
            if(!MatrixTools.getDimensions(forecast).equalsIgnoreCase(MatrixTools.getDimensions(test))){
                System.out.println(getModel());
                throw new IllegalArgumentException("both matrices need to have the same dimensions the dimensions are :"+ MatrixTools.getDimensions(forecast) +" and "+MatrixTools.getDimensions(test));
            }
            return m.error(this);
        }
        public double error_truncated(Metric m, int off_set){
            return m.error_truncated(this,off_set);
        }
        public RealVector error_vec(Metric m){
            return m.error_vec(this);
        }

        public RealVector error_truncated_vec(Metric m, int off_set) {
            return m.error_truncated_vec(this, off_set);
        }

        public Output copy() {
            Output o = new Output();
            o.test = test.copy();
            o.train = train.copy();
            o.forecast = forecast.copy();
            o.runTime = runTime;
            return o;
        }
        public Output sub_copy(int index){
            Output o = new Output();
            o.test = test.getRowMatrix(index);
            o.train = train.getRowMatrix(index);
            o.forecast = forecast.getRowMatrix(index);
            o.runTime = runTime;
            return o;

        }
        public Output sub_copy(int start, int len){
            Output o = new Output();
            o.test = test.getSubMatrix(start,start+len-1,0,test.getColumnDimension()-1);
            o.train = train.getSubMatrix(start,start+len-1,0,train.getColumnDimension()-1);
            o.forecast = forecast.getSubMatrix(start,start+len-1,0,forecast.getColumnDimension()-1);
            o.runTime = runTime;
            return o;

        }

        public void store(String output_path) throws IOException {
            BufferedWriter writer = new BufferedWriter(new FileWriter(new File(output_path)));
            writer.write(this.toString());
            writer.flush();

        }

        public Output normalize() {
            RealMatrix ts = MatrixUtils.createRealMatrix(train.getRowDimension(),train.getColumnDimension()+test.getColumnDimension());
            ts.setSubMatrix(train.getData(),0,0);
            ts.setSubMatrix(test.getData(),0,train.getColumnDimension());
            ArrayList<RealMatrix> ret = MatrixTools.normalize(ts,forecast);
            train = ret.get(0).getSubMatrix(0,train.getRowDimension()-1,0,train.getColumnDimension()-1);
            test = ret.get(0).getSubMatrix(0,train.getRowDimension()-1,train.getColumnDimension(),ts.getColumnDimension()-1);
            forecast = ret.get(1);
            return this;
        }
    }

    public class MultiOutput{
        RealMatrix train;
        RealMatrix test;
        double runtime;
        HashMap<Integer,RealMatrix> forecasts = new HashMap<>();

        public MultiOutput(Output o) {
            train =  o.train;
            test = o.test;
        }

        public MultiOutput(HashMap<Integer, Output> outputs2) {
        }

        public RealMatrix getTrain() {
            return train;
        }
        public RealMatrix getTest(){
            return test;
        }

        public void plotComparrisons(){
            forecasts.forEach((k,v)->new Output(train,test,v).plotComparrison(true,k.toString()));
        }

        public void setRunTime(double runtime) {
            this.runtime = runtime;
        }

        public void add_horizon(int horizon, RealMatrix realMatrix) {
            forecasts.put(horizon,realMatrix);
        }

        public HashMap<Integer,Double> errors(Metric m) {
            HashMap<Integer,Double> ret = new HashMap<>();
            forecasts.forEach((k,v) -> ret.put(k,new Output(train,test,v).error(m)));
            return ret;
        }
        public Graph error_graph(Metric m){
            return error_graph(m,"");
        }
        public Graph error_graph(Metric m, String dataSetName){
            Graph graph = new Graph(dataSetName, "Error", "horizon", m.toString(),false);
            HashMap<Integer,Double> ret = errors(m);
            ret.forEach((k,v)->graph.add(TSModel.this,k,v));
            return graph;
        }

        public RealMatrix at_horizon(int i) {
            return forecasts.get(i);
        }

        public Output output_at_horizon(int j) {
            return new Output(train,test,at_horizon(j),runtime);
        }

        public double error(int horizon,Metric metric) {
            return new Output(train,test,at_horizon(horizon),runtime).error(metric);
        }

        public double getRunTime() {
            return runtime;
        }
    }

    protected static Output decode_output(String s,TSModel model) {
        Output o = model.new Output();
        String[] matrices = s.split("%%%");
        o.setRunTime(new Double(matrices[0]));
        o.train = decode_matrix(matrices[1]);
        o.test = decode_matrix(matrices[2]);
        o.forecast = decode_matrix(matrices[3]);
        return o;

    }
    public static Output decode_output(String s) {
        String[] matrices = s.split("%%%");
        String modelname = matrices[4];
        modelname = modelname.replaceAll("%","");
        Output o = null;
        try {
            Class<? extends TSModel> clazz = (Class<? extends TSModel>) Class.forName(modelname);
            Constructor<? extends TSModel> ctor = clazz.getConstructor();
            TSModel model = ctor.newInstance();
            o = model.new Output();
        } catch (InstantiationException e) {
            e.printStackTrace();
            System.exit(-1);
        } catch (InvocationTargetException e) {
            e.printStackTrace();
            System.exit(-1);
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
            System.exit(-1);
        } catch (IllegalAccessException e) {
            e.printStackTrace();
            System.exit(-1);
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
            System.exit(-1);
        }
        o.setRunTime(new Double(matrices[0]));
        o.train = decode_matrix(matrices[1]);
        o.test = decode_matrix(matrices[2]);
        o.forecast = decode_matrix(matrices[3]);
        return o;

    }
    public static Output load_output(String path) throws FileNotFoundException {
        File f = new File(path);
        BufferedReader reader = new BufferedReader(new FileReader(f));
        String encoded = reader.lines().collect(Collectors.joining(System.lineSeparator()));
        return decode_output(encoded);

    }


    protected static RealMatrix decode_matrix(String s) {
        if (s == "!") {
            return null;
        }
        String[] rows = s.split("%");
        ArrayList<ArrayList<Double>> temp = new ArrayList<>();
        ArrayList<Double> temptemp = new ArrayList<>();
        for (String row : rows) {
            if (row.length() == 0) {
                continue;
            }
            temptemp = new ArrayList<>();
            String[] entries = row.split(" ");
            for (String entry : entries) {
                if (entry.length() == 0) {
                    continue;
                }
                try {
                    temptemp.add(new Double(entry));
                } catch (NumberFormatException e) {
                    temptemp.add(Double.NaN);
                }
            }
            temp.add(temptemp);
        }
        RealMatrix r = MatrixUtils.createRealMatrix(temp.size(), temptemp.size());
        for (int i = 0; i < temp.size(); i++) {
            for (int j = 0; j < temptemp.size(); j++) {
                r.setEntry(i, j, temp.get(i).get(j));
            }
        }
        return r;
    }



}
