package thesis.Tools;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.LogAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.shade.protobuf.MapEntry;
import thesis.Plotting.Plotter;
import thesis.TSModel;

import javax.swing.*;
import java.io.*;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.*;

public class Graph {

    private String debugPath = "Results\\Debug\\";

    //takes as input a encoding of the graph:
    //val1: x-values
    //fileName: name of the dataset

    private boolean shapes;
    private boolean log_scale_Y;
    private int log_y_base = 10;
    private boolean log_scale_x;
    private int log_scale_x_base = 2;


    private IndexSet indexs;
    private HashMap<String,ArrayList<Double>> ys = new HashMap<>();


    private String fileName;
    private String valueString;

    private String titel;


    private Double last_index;

    private double runtime = Double.NaN;


    public Graph(){
    }
    public Graph(IndexSet indxs){
        this.indexs = indxs;
    }

    public Graph(String datasetName) {
        fileName = datasetName;
    }
    public Graph(String fileName,String titel, HashMap<String,ArrayList<Double>> ys, IndexSet indsexs, String valueString){
        this.fileName = fileName;
        this.titel = titel;
        this.ys = ys;
        this.indexs = indsexs;
        this.valueString = valueString;
    }
    public Graph(String datasetName, String titel, String indexString, String metricString) {
        valueString = metricString;
        indexs = new IndexSet(indexString);
        fileName = datasetName;
        this.titel = titel;
        this.shapes = true;
    }
    public Graph(String datasetName, String titel, String indexString, String metricString,boolean shapes) {
        valueString = metricString;
        indexs = new IndexSet(indexString);
        fileName = datasetName;
        this.titel = titel;
        this.shapes = shapes;
    }
    public Graph(String datasetName, String titel, String indexString, String metricString, IndexSet set) {
        valueString = metricString;
        indexs = new IndexSet(indexString);
        fileName = datasetName;
        this.titel = titel;
        this.shapes = false;
        this.indexs = set;
    }
    public Graph(IndexSet indxs, TimeSeries t){
        indexs = indxs;
        RealMatrix asMatrix = t.toMatrix();
        ArrayList<Double> temp;
        ys = new HashMap<>();
        for(int i = 0; i<asMatrix.getRowDimension();i++){
            temp = new ArrayList<>();
            for(int j = 0; j<asMatrix.getColumnDimension();j++){
                temp.add(asMatrix.getEntry(i,j));
            }
            ys.put(new Integer(i).toString(),temp);
        }

    }
    public Graph(TimeSeries t){
        shapes = true;
        indexs = new IndexSet("timeStep", 0, t.length());
        RealMatrix asMatrix = t.toMatrix();
        ArrayList<Double> temp;
        ys = new HashMap<>();
        for(int i = 0; i<asMatrix.getRowDimension();i++){
            temp = new ArrayList<>();
            for(int j = 0; j<asMatrix.getColumnDimension();j++){
                temp.add(asMatrix.getEntry(i,j));
            }
            ys.put(new Integer(i).toString(),temp);
        }

    }

    public static Graph merge(Graph error_graph, Graph all_horizon) {
        if(error_graph.ys.isEmpty()){
            error_graph.set(all_horizon);
        }
        else{
            error_graph = error_graph.merge(all_horizon);
        }
        return error_graph;
    }

    public static Graph all_horizon_graph(TSModel.Output o, Metric m){
        return all_horizon_graph_aggregated(o,m,1,false);
    }
    public static Graph all_horizon_graph_aggregated(String dataset_name, String method,TSModel.Output o, Metric m, int step_size, boolean truncate){
        RealMatrix test = o.getTest().copy();
        RealMatrix forecast = o.getForecast().copy();
        Graph g = new Graph();
        g.indexs = new IndexSet("Horizon",step_size,o.getTest().getColumnDimension(),step_size);
        if(g.indexs.indexs().size()<=100)
            g.shapes = true;
        g.valueString = "Error";
        g.titel = "Vary Horizon "+ dataset_name;
        g.ys = new HashMap();
        g.ys.put(method,new ArrayList<>());
        double[] temp = new double[step_size];
        int count = 0;
        if(!truncate) {
            for (int i = 0; i < test.getColumnDimension(); i++) {
                o.setForecast(forecast.getSubMatrix(0, forecast.getRowDimension() - 1, 0, i));
                o.setTest(test.getSubMatrix(0, test.getRowDimension() - 1, 0, i));
                if(m instanceof Metric.SRW_RMSE) {
                    temp[count++] = ((Metric.SRW_RMSE)m).error_auto_period(dataset_name,o);
                }
                else{
                    temp[count++] = o.error(m);
                }

                if (count == step_size || i == test.getColumnDimension() - 1) {
                    g.ys.get(method).add(ArrayUtils.sum(temp, 0, count) / count);
                    count = 0;
                }

            }
        }
        else {
            for (int i = 0; i < test.getColumnDimension(); i += step_size) {
                int len = i + step_size >= test.getColumnDimension() ? test.getColumnDimension() - i : step_size;
                o.setForecast(forecast.getSubMatrix(0, forecast.getRowDimension() - 1, 0, i + len - 1));
                o.setTest(test.getSubMatrix(0, test.getRowDimension() - 1, 0, i + len - 1));
                if(m instanceof Metric.SRW_RMSE) {
                    g.ys.get(method).add(((Metric.SRW_RMSE)m).error_trunc_auto_period(dataset_name,o, i));
                }
                else{
                    g.ys.get(method).add(o.error_truncated(m, i));
                }

            }
        }
        o.setForecast(forecast);
        o.setTest(test);
        return g;
    }
    public String to_latex_heatmap(){
        StringBuilder sb = new StringBuilder();
        //header
        sb.append("\\begin{table}[ht]\n");

        sb.append(" \\definecolor{lt2}{rgb}{0.9,0.9,0.9}            %<  0.2 \n");
        sb.append("\\definecolor{lt4}{rgb}{0.8, 0.8, 0.8}    %<  0.4 \n");
        sb.append("\\definecolor{lt6}{rgb}{0.7, 0.7, 0.7}    %<  0.6 \n");
        sb.append("\\definecolor{lt8}{rgb}{0.5, 0.5, 0.5}    %<  0.8 \n");
        sb.append("\\definecolor{lt10}{rgb}{0.375,0.375,0.375}      %<  1 \n");
        sb.append("\\definecolor{lt12}{rgb}{0.25, 0.25, 0.25}    %<  1.2 \n");
        sb.append("\\definecolor{lt14}{rgb}{0.1, 0.1, 0.1}    %<  1.4 \n");
        sb.append("\\definecolor{gt14}{rgb}{0.0, 0.0, 0.0}    %>  1.4 \n");

        sb.append("\\begin{center}\n");
        sb.append("\\begin{tabular}{|c c c c c c c c c c c| c c c c c c}\n");
        String[] method_ordering_arr = {"thesis.Models.RandomForestND","thesis.Models.RandomForest1D"
                , "thesis.Models.GBMND","thesis.Models.GBM1D","thesis.Models.TCNModelND","thesis.Models.TCNModel1D","thesis.Models.TransformerModelND",
                "thesis.Models.TransformerModel1D","thesis.Models.BlockRNNND","thesis.Models.BlockRNN1D","thesis.Models.N_BEATS","thesis.Models.BeatlexAltND",
                "thesis.Models.BeatlexAlt1D", "thesis.Models.FFT","thesis.Models.SES","thesis.Models.TBATS","thesis.Models.ME","thesis.Models.ARIMA","thesis.Models.trmf",
                "thesis.Models.ThetaForecaster","thesis.Models.OTS","thesis.Models.Prophet"
        };
        //this is used to add the legend to the plot
        String[] addon = {"&&&&&","&&&&&","&&&&&","&&&&&","&&&&&","&&&&&","&&&&&","&&&\\multicolumn{3}{l}{\\textbf{ n-RMSE}}","&&&&\\cellcolor{lt2}&$\\leq 0.2$","&&&&\\cellcolor{lt4}&$\\leq 0.4$",
                "&&&&\\cellcolor{lt6}&$\\leq 0.6$","&&&&\\cellcolor{lt8}&$\\leq 0.8$", "&&&&\\cellcolor{lt10}&$\\leq 1$", "&&&&\\cellcolor{lt12}&$\\leq 1.2$", "&&&&\\cellcolor{lt14}&$\\leq 1.4$",
                "&&&&\\cellcolor{gt14}&$\\geq 1.4$","&&&&&","&&&&&","&&&&&","&&&&&","&&&&&","&&&&&"};
        ArrayList<String> method_ordering = new ArrayList<>(Arrays.asList(method_ordering_arr));


        //adding the step sizes
        sb.append("&");
        get_indexs().indexs().forEach((v)->{
            sb.append(new Integer(new Double(v).intValue()).toString()+"&");
        });
        sb.deleteCharAt(sb.length()-1);
        sb.append("\\\\ \n");


        RealMatrix result_matrix = MatrixUtils.createRealMatrix(method_ordering.size(),ys.values().iterator().next().size());
        ys.forEach((method,errors)->{
            int row = method_ordering.indexOf(method);
            //annoyance due to GBM and R_F
            if(row == -1){
                row = method_ordering.indexOf(method+"1D");
            }
            if(row!=-1) {
                System.out.println("pepe");
                for (int i = 0; i < errors.size(); i++) {
                    result_matrix.setEntry(row, i, round(errors.get(i), 2));
                }
            }
        });
        for(int i = 0; i<method_ordering.size();i++){
            String method = method_ordering.get(i).substring(method_ordering.get(i).lastIndexOf(".")+1);
            sb.append(method+"&");
            for(int j = 0; j<result_matrix.getColumnDimension();j++){
                if(result_matrix.getEntry(i,j)<0.2){
                    sb.append("\\cellcolor{lt2}&");
                }
                else if(result_matrix.getEntry(i,j)<0.4){
                    sb.append("\\cellcolor{lt4}&");
                }
                else if(result_matrix.getEntry(i,j)<0.6){
                    sb.append("\\cellcolor{lt6}&");
                }
                else if(result_matrix.getEntry(i,j)<0.8){
                    sb.append("\\cellcolor{lt8}&");
                }
                else if(result_matrix.getEntry(i,j)<1){
                    sb.append("\\cellcolor{lt10}&");
                }
                else if(result_matrix.getEntry(i,j)<1.2){
                    sb.append("\\cellcolor{lt12}&");
                }
                else if(result_matrix.getEntry(i,j)<1.4){
                    sb.append("\\cellcolor{lt14}&");
                }
                else if(result_matrix.getEntry(i,j)>01.4){
                    sb.append("\\cellcolor{gt14}&");
                }
            }
            sb.deleteCharAt(sb.length()-1);
            sb.append(addon[i]);
            sb.append( "\\\\ \n");

        }

        sb.append("\\end{tabular}\n");
        sb.append("\\end{center}\n");
        sb.append("\\end{table}\n");


        return sb.toString();


    }
    public static Graph all_horizon_graph_aggregated(TSModel.Output o, Metric m, int step_size, boolean truncate){
        return all_horizon_graph_aggregated("",o.getModel(),o,m,step_size,truncate);
    }
    public static Graph all_horizon_log_aggregated(TSModel.Output o, Metric m){
        RealMatrix test = o.getTest().copy();
        RealMatrix forecast = o.getForecast().copy();
        Graph g = new Graph();
        g.indexs = new IndexSet("Horizon");
        g.valueString = "Error";
        g.titel = "";
        g.shapes = true;
        g.ys = new HashMap();
        g.ys.put(o.getModel(),new ArrayList<>());
        int cur_index = 0;
        while(cur_index<test.getColumnDimension()){

            int step_size = step_size(cur_index+1);
            if(cur_index+step_size-1>=test.getColumnDimension()){
                step_size = test.getColumnDimension()-cur_index;
            }
            o.setForecast(forecast.getSubMatrix(0,forecast.getRowDimension()-1,cur_index,cur_index+step_size-1));
            o.setTest(test.getSubMatrix(0,test.getRowDimension()-1,cur_index,cur_index+step_size-1));
            g.indexs.indexs().add((double)(cur_index+1));
            g.ys.get(o.getModel()).add(o.error(m));
            cur_index += step_size;

        }
        o.setTest(test);
        o.setForecast(forecast);

        return g;

    }
    private static int step_size(int cur){
        int os = (int)Math.floor(Math.log10(cur));
        StringBuilder s = new StringBuilder();
        s.append("1");
        for(int i = 1; i<=os;i++){
            s.append("0");
        }
        return new Integer(s.toString());
    }

    private void set(Graph graph) {
        shapes = graph.shapes;
        indexs = graph.indexs;
        ys = graph.ys;
        fileName = graph.fileName;
        valueString = graph.valueString;
        titel = graph.titel;
        last_index = graph.last_index;

    }

    public Graph getSubGraph(String key){
        HashMap<String,ArrayList<Double>> h = new HashMap<>();
        h.put(key,ys.get(key));
        return new Graph(fileName,titel,h,indexs,valueString);
    }
    public void add(TSModel model, double index, double value){
        if (!ys.containsKey(model.toString())){
            ys.put(model.toString(),new ArrayList<>());
        }
        ys.get(model.toString()).add(value);

        if (last_index ==null || index!=last_index) {
            indexs.indexs().add(index);
            last_index = index;
        }


    }
    public void add(String key, double index, double value){
        if (!ys.containsKey(key)){
            ys.put(key,new ArrayList<>());
        }
        ys.get(key).add(value);

        if (last_index ==null || index!=last_index) {
            indexs.indexs().add(index);
            last_index = index;
        }


    }
    public String to_sns_heat_map(){
        String returner = "df = pd.DataFrame(data=[";
        ArrayList<String> names = new ArrayList<>();
        int index = 0;
        Double[][] matrix = new Double[ys.size()][];
        for (Map.Entry<String, ArrayList<Double>> entry : ys.entrySet()) {
            String name = entry.getKey();
            ArrayList<Double> values = entry.getValue();
            names.add(name);
            matrix[index++] = values.toArray(new Double[]{});

        }
        for(int j = 0; j<matrix.length;j++){
            Double[] row = matrix[j];
            returner += "[";
            for(int i = 0;i<row.length;i++){
                returner+= row[i].toString();
                if(i<row.length-1)
                    returner+= ", ";
            }
            returner += "]";
            if(j<matrix.length-1)
                returner+=",";
        }
        returner+= "], index=[";
        for(int i = 0;i<names.size();i++){
            String name = names.get(i);
            name = name.substring(name.lastIndexOf(".")+1);
            returner+="\""+name+"\"";
            if(i<names.size()-1)
                returner+=", ";
        }
        returner+= "], columns=[";
        for(int i = 0; i<indexs.indexs().size();i++){
            returner+= indexs.indexs().get(i);
            if(i<indexs.indexs().size()-1){
                returner+= ", ";
            }
        }
        returner+="])\nax = sns.heatmap(df, annot=True)\n ax.figure.tight_layout()\n  plt.show()";
        return returner;

    }


    public void add(String s, RealMatrix ys) {
        ArrayList<Double> arr;
        for(int i = 0; i<ys.getRowDimension();i++){
            arr = new ArrayList<>();
            for(int j = 0; j<ys.getColumnDimension();j++){
                arr.add(ys.getEntry(i,j));
            }
            this.ys.put(s+" "+i,arr);
        }

    }

    public void write_to_file(String name, String dataSetName, String forecastingMethod, String mode, Double value) throws FileNotFoundException {

        File file;
        StringBuilder sb;

        String path = graphPath(name,dataSetName,forecastingMethod,mode,value);
        String[] temp;
        String fileName = (temp=path.split("\\\\"))[temp.length-1];
        file = new File(path.replaceAll(fileName,""));
        file.mkdirs();

        file = new File(path);
        try (PrintWriter writer = new PrintWriter(file)) {
            sb = new StringBuilder();
            sb.append(indexs.toString());
            sb.append("###");
            sb.append(ysToString());
            System.out.println(path);
            writer.write(sb.toString());
            writer.flush();
        }
        catch (IOException e){
            e.printStackTrace();
        }

    }
    static String graphPath(String name, String dataSetName, String forecastingMethod, String mode, Double value){
        return"Results\\"
                +name+
                "\\"
                +getName(dataSetName)+
                "\\"
                +forecastingMethod+
                "\\"
                +mode+
                (value == null ? "": round(value,2))+
                ".txt";

    }
    private static String getName(String dataSetName){
        String[] bananaSplit = dataSetName.split("\\\\");
        return bananaSplit[bananaSplit.length-1].replaceAll(".csv","");
    }
    static Double round(Double d, int precise) {
        if(d.isNaN()){
            return d;
        }
        BigDecimal bigDecimal = BigDecimal.valueOf(d);
        bigDecimal = bigDecimal.setScale(precise, RoundingMode.HALF_DOWN);
        return bigDecimal.doubleValue();
    }


    public static Graph read_from_file(String filePath) throws IOException {
        Graph ret = new Graph();
        String[] split = filePath.split("\\\\");
        ret.fileName = split[2];
        ret.titel = split[3];
        ret.valueString = split.length==5 ? "seconds" : split[4];


        String row;
        StringBuilder fileContent = new StringBuilder();
        BufferedReader csvReader = new BufferedReader(new FileReader(filePath));
        while ((row = csvReader.readLine()) != null) {
            fileContent.append(row+"\n");
        }
        csvReader.close();
        String[] temp = fileContent.toString().split("###");
        ret.indexs = IndexSet.parse_indexset(temp[0]);
        ret.ys = parseYs(temp[1]);
        return ret;
    }
    public static Graph read_from_file(String name, String dataSetName, String forecastingMethod, String mode, double value) throws IOException {
        String path = graphPath(name,dataSetName,forecastingMethod,mode,value);
        return read_from_file(path);

    }

    public static Graph read_from_file(String name, String dataSetName, String forecastingMethod, String mode) throws IOException {
        String path = graphPath(name,dataSetName,forecastingMethod,mode,null);
        return read_from_file(path);

    }

    public void set_log_scale_y(boolean flag){
        log_scale_Y = flag;
    }
    public void set_log_base_y(int base){
        log_scale_Y = true;
        log_y_base = base;
    }
    public void set_log_base_x(int base){
        log_scale_x = true;
        log_scale_x_base = base;
    }
    public void set_log_scale_x(boolean flag){
        log_scale_x = flag;
    }




    public void plot() {
        XYSeriesCollection image = new XYSeriesCollection();
        ys.forEach((k,v) -> image.addSeries(createSeries(k,v)));

        Plotter awt= new Plotter(titel+" "+fileName,image,indexs.getDescription(),valueString,shapes,log_scale_Y,log_scale_x_base,log_scale_x,log_scale_x_base);
        awt.pack();
        awt.setVisible(true);
    }
    public void plot(String mode){
        if(mode.equalsIgnoreCase("log x")){
            XYSeriesCollection image = new XYSeriesCollection();
            ys.forEach((k,v) -> image.addSeries(createSeries(k,v)));
            NumberAxis yAxis = new NumberAxis(valueString);
            yAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
            LogAxis xAxis = new LogAxis(indexs.getDescription());
            xAxis.setBase(10);
            xAxis.setStandardTickUnits(NumberAxis.createIntegerTickUnits());
            XYPlot plot = new XYPlot(image,
                    xAxis, yAxis, new XYLineAndShapeRenderer(true, true));
            JFreeChart chart = new JFreeChart(
                    "Chart", JFreeChart.DEFAULT_TITLE_FONT, plot, true);
            JFrame frame = new JFrame("LogAxis Test");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setContentPane(new ChartPanel(chart));
            frame.pack();
            frame.setVisible(true);

        }
    }
    public void plotPairwiseComparison(int cap) {
            int count = 0;
            XYSeriesCollection image = new XYSeriesCollection();
            for (Map.Entry<Tuple<String,String>, Tuple<ArrayList<Double>,ArrayList<Double>>> entry : pair(ys)) {
                Tuple<String,String> k = entry.getKey();
                Tuple<ArrayList<Double>,ArrayList<Double>> v = entry.getValue();
                image = new XYSeriesCollection();
                image.addSeries(createSeries(k.getVal1(), v.getVal1()));
                image.addSeries(createSeries(k.getVal2(), v.getVal2()));

                Plotter awt= new Plotter(titel,image,indexs.getDescription(),valueString,shapes);
                awt.pack();
                awt.setVisible(true);
                if(++count>=cap)
                    break;

            }


    }

    private Iterable<? extends Map.Entry<Tuple<String,String>, Tuple<ArrayList<Double>, ArrayList<Double>>>> pair(HashMap<String, ArrayList<Double>> ys) {
        HashMap<Tuple<String,String>,Tuple<ArrayList<Double>, ArrayList<Double>>> pairs = new HashMap<>();

        LinkedList<String> keys = new LinkedList<>(ys.keySet());
        int curInteger;
        String key;
        String validationString;
        String testString;
        while (!keys.isEmpty()){
            key = keys.peek();

            curInteger = new Integer(key.contains("validation ") ? key.replace("validation ","") : key.replace("test ",""));
            validationString = "validation "+curInteger;
            testString = "test "+curInteger;

            pairs.put(new Tuple<>(validationString,testString),new Tuple<>(ys.get(validationString),ys.get(testString)));
            keys.remove(testString);
            keys.remove(validationString);
        }
        return pairs.entrySet();

    }

    private XYSeries createSeries(String k, ArrayList<Double> v) {
        int count = 0;
        XYSeries ret = new XYSeries(k);
        //Collections.sort(indexs.indexs());
        //System.out.println(Arrays.toString(indexs.indexs().stream().toArray(Double[]::new)));
        for (Double d : v){
            ret.add(indexs.indexs().get(count++),d);
        }
        return ret;
    }


    public String toString(){
        return "dataSet: "+ fileName+"\nmetric: "+valueString+"\nIndexSet: "+indexs.toString()+"\n"+ysToString();
    }
    private String ysToString(){
        StringBuilder ret = new StringBuilder();
        ys.forEach((k,v) -> {
            ret.append(k+", ");
            ret.append(Arrays.toString(ys.get(k).stream().toArray(Double[]::new)));
            ret.append("\n");
        });
        return ret.toString().replaceAll("\\[","").replaceAll("\\]","");

    }
    public static HashMap<String,ArrayList<Double>> parseYs(String ysEncoded){
        HashMap<String,ArrayList<Double>> ys = new HashMap<>();
        String[] lines = ysEncoded.split("\n");
        ArrayList<Double> array;
        String[] lineSplit;
        String key;
        for(String line : lines){
            array = new ArrayList<>();
            lineSplit = line.split(", ");
            key = lineSplit[0];
            String[] temp = new String[lineSplit.length-1];
            System.arraycopy(lineSplit,1,temp,0,temp.length);
            lineSplit = temp;
            for (String x : lineSplit) {
                try {
                    array.add(new Double(x));
                }
                catch (NumberFormatException e){
                    array.add(new Double(Double.NaN));
                }
            }
            ys.put(key,array);
        }
        return ys;
    }


    public Graph merge(Graph g) {
        g.ys.forEach((k,v)->ys.put(k,v));
        return this;
    }

    public void set_indexs(IndexSet indexs){
        this.indexs = indexs;
    }
    public void set_value_string(String s){
        valueString = s;
    }


    public void dots(boolean b) {
        shapes = b;
    }


    public TSModel.Output to_output(double split,TSModel model) {
        TimeSeries temp;
        TSModel.Output o = model.new Output(new TimeSeries(ys.get("test")).toMatrix(),split);
        o.setForecast((temp = new TimeSeries(ys.get("validation"))).getSubSeries(temp.length()-o.getTest().getColumnDimension(),o.getTest().getColumnDimension()).toMatrix());
        return o;
    }

    public String to_latex_table(int prec){
        StringBuilder sb = new StringBuilder();
        for(Map.Entry<String, ArrayList<Double>> entry : ys.entrySet()){
            sb.append(entry.getKey());
            sb.append("&");
            for(int i = 0; i<entry.getValue().size();i++){
                if(i == entry.getValue().size()-1){
                    sb.append(round(entry.getValue().get(i), prec));
                    sb.append("\\\\");
                    sb.append("\n");
                    sb.append("\\hline");
                    sb.append("\n");
                }
                else {
                    sb.append(round(entry.getValue().get(i), prec));
                    sb.append("&");
                }
            }
        }
        return sb.toString();
    }

    public HashMap<String, ArrayList<Double>> get_ys() {
        return ys;
    }

    public IndexSet get_indexs() {
        return indexs;
    }

    public void set_titel(String titel) {
        this.titel = titel;
    }

    public void set_data_set(String data_path) {
        this.fileName = data_path;
    }


}
