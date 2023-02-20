package thesis.Tools;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import thesis.Models.ARIMA;
import thesis.Models.SeasonalRandomWalk;
import thesis.TSModel;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

public abstract class Metric{
    public enum Collapse{
        MEAN,
        MEDIAN;

        public double collapse(RealVector r){
            double tot = 0;
            int count = 0;
            if(this == MEAN){
                for(double v : r.toArray()){
                    tot += v;
                    count++;
                }
                return tot/count;

            }
            else if(this == MEDIAN){
                double[] vec = r.toArray();
                Arrays.sort(vec);
                double median;
                if (vec.length % 2 == 0)
                    median = (vec[vec.length/2] + vec[vec.length/2 - 1])/2;
                else
                    median = vec[vec.length/2];
                return median;
            }
            else{
                throw new Error("How is this even possible");
            }
        }
    }
    protected Collapse collapse = Collapse.MEAN;

    public void set_collapser(Collapse c){
        collapse = c;
    }

    public double error(TSModel.Output o){
        RealVector v = error_vec(o);
        return collapse.collapse(v);
    }
    public double error_truncated(TSModel.Output o, int off_set) {return collapse.collapse(error_truncated_vec(o, off_set));}
    public static Metric[] metrics() {
        return new Metric[]{new RMSE()};
    }

    public RealVector error_vec(TSModel.Output o){
        return error_truncated_vec(o,0);
    }
    public abstract RealVector error_truncated_vec(TSModel.Output o, int off_set);

    public abstract String getDescription();

    public static class MWOES extends Metric{

        @Override
        public RealVector error_truncated_vec(TSModel.Output o, int off_set) {
            RealMatrix test = o.getTest();
            RealMatrix forecast = o.getForecast();
            RealVector ret = MatrixUtils.createRealVector(new double[test.getRowDimension()]);
            int count = 0;
            double tot = 0;
            for(int i = 0; i<test.getRowDimension();i++){
                for(int j = off_set;j<test.getColumnDimension();j++){
                    tot += test.getEntry(i,j)<forecast.getEntry(i,j) ? 1 : 0;
                    count++;
                }
                if(count!=0)
                    ret.setEntry(i,tot/count);
                count = 0;
                tot = 0;
            }
            return ret;
        }

        @Override
        public String getDescription() {
            return "MWOES";
        }
    }
    public static class MWUES extends Metric{

        @Override
        public RealVector error_truncated_vec(TSModel.Output o, int off_set) {
            RealMatrix test = o.getTest();
            RealMatrix forecast = o.getForecast();
            RealVector ret = MatrixUtils.createRealVector(new double[test.getRowDimension()]);
            int count = 0;
            double tot = 0;
            for(int i = 0; i<test.getRowDimension();i++){
                for(int j = off_set;j<test.getColumnDimension();j++){
                    tot += test.getEntry(i,j)>forecast.getEntry(i,j) ? 1 : 0;
                    count++;
                }
                if(count!=0)
                    ret.setEntry(i,tot/count);
                count = 0;
                tot = 0;
            }
            return ret;
        }

        @Override
        public String getDescription() {
            return "MWUES";
        }
    }
    public static class MWOAS extends Metric{

        @Override
        public RealVector error_truncated_vec(TSModel.Output o, int off_set) {
            RealVector esti = o.error_vec(new MWOES());
            RealMatrix test = o.getTest();
            RealMatrix forecast = o.getForecast();
            RealVector ret = MatrixUtils.createRealVector(new double[test.getRowDimension()]);
            int count = 0;
            double tot = 0;
            for(int i = 0; i<test.getRowDimension();i++){
                if(esti.getEntry(i) == 0) {
                    ret.setEntry(i,0);
                    continue;
                }
                for(int j = off_set;j<test.getColumnDimension();j++){
                    if(test.getEntry(i,j)==0){
                        throw new Error("0 values are not allowed in the test set");
                    }
                    tot += test.getEntry(i,j)<forecast.getEntry(i,j) ? (forecast.getEntry(i,j) - test.getEntry(i,j))/Math.abs(test.getEntry(i,j)) : 0;
                    count++;
                }
                if(count!=0)
                    ret.setEntry(i,tot/(count*esti.getEntry(i)));
                count = 0;
                tot = 0;
            }
            return ret;
        }

        @Override
        public String getDescription() {
            return "MWOAS";
        }
    }
    public static class SMWOAS extends Metric{

        @Override
        public RealVector error_truncated_vec(TSModel.Output o, int off_set) {
            RealVector esti = o.error_vec(new MWOES());
            RealMatrix test = o.getTest();
            RealMatrix forecast = o.getForecast();
            RealVector ret = MatrixUtils.createRealVector(new double[test.getRowDimension()]);
            int count = 0;
            double tot = 0;
            for(int i = 0; i<test.getRowDimension();i++){
                if(esti.getEntry(i) == 0) {
                    ret.setEntry(i,0);
                    continue;
                }
                for(int j = off_set;j<test.getColumnDimension();j++){
                    if(test.getEntry(i,j)==0&&forecast.getEntry(i,j)==0){
                        ret.setEntry(i,Double.NaN);
                        break;
                    }
                    tot += forecast.getEntry(i,j)>test.getEntry(i,j) ? Math.abs(forecast.getEntry(i,j) - test.getEntry(i,j))/((Math.abs(test.getEntry(i,j))+Math.abs(forecast.getEntry(i,j)))/2) : 0;
                    count++;
                }
                if(count!=0)
                    ret.setEntry(i,esti.getEntry(i) == 0 ? 0 : tot/(count*esti.getEntry(i)));
                count = 0;
                tot = 0;
            }
            return ret;
        }

        @Override
        public String getDescription() {
            return "SMWOAS";
        }
    }
    //normalizes the RMSE by the RMSE of the RW model
    public static class SRW_RMSE extends  Metric{
        SeasonalRandomWalk s_r_w;
        private HashMap<String,Integer> data_period_mapping = dataset_period_mapping();

        public SRW_RMSE(int period){
            s_r_w = new SeasonalRandomWalk(period);

        }

        public SRW_RMSE(){


        }

        public HashMap<String,Integer> dataset_period_mapping(){
            HashMap<String, Integer> h = new HashMap<>();
            h.put("econometric_indicator#103",1);
            h.put("econometric_indicator#106",1);
            h.put("econometric_indicator#67",1);
            h.put("econometric_indicator#77",1);
            h.put("electricity_consumption#245",4);
            h.put("electricity_consumption#147",4);
            h.put("electricity_consumption#246",4);
            h.put("electricity_consumption#268",4);
            h.put("electricity_consumption#130",28);
            h.put("solar_production#0",4);
            h.put("ATM_withdraw#110",7);
            h.put("ATM_withdraw#109",7);

            return h;

        }
        public double error_auto_period(String dataset, TSModel.Output o){
            if(data_period_mapping.containsKey(dataset)){
                s_r_w = new SeasonalRandomWalk(data_period_mapping.get(dataset));
            }
            else{
                s_r_w = new SeasonalRandomWalk(1);
            }
            return super.error(o);

        }

        public double error_trunc_auto_period(String dataset, TSModel.Output o, int offset){
            if(data_period_mapping.containsKey(dataset)){
                s_r_w = new SeasonalRandomWalk(data_period_mapping.get(dataset));
            }
            else{
                s_r_w = new SeasonalRandomWalk(1);
            }
            return super.error_truncated(o,offset);

        }


        @Override
        public RealVector error_truncated_vec(TSModel.Output o, int off_set) {
            TSModel.Output out2 = s_r_w.test_accuracy(o.getTrain(),o.getTest());
            RealVector o2_error_vec = out2.error_truncated_vec(new RMSE(),off_set);
            RealVector o_error_vec = o.error_truncated_vec(new RMSE(),off_set);
            RealVector quotient = o_error_vec.ebeDivide(o2_error_vec);
            return quotient;
        }

        @Override
        public String getDescription() {
            return "SRW-RMSE";
        }

        @Override
        public String toString() {
            return "SRW-RMSE";
        }
    }
    public static class ANRMSE extends Metric{
        //RMSE normalized by level

        public ANRMSE() {

        }
        public String toString(){
            return "n-RMSE";
        }


        @Override
        public RealVector error_truncated_vec(TSModel.Output o, int off_set) {
            RealMatrix test = o.getTest();
            RealMatrix forecast = o.getForecast();
            RealMatrix temp = MatrixUtils.createRealMatrix(forecast.getRowDimension(),forecast.getColumnDimension());
            RealVector rowAve = MatrixUtils.createRealVector(new double[temp.getRowDimension()]);
            RealVector y_ave = MatrixUtils.createRealVector(new double[temp.getRowDimension()]);
            for(int i = 0; i<temp.getRowDimension();i++){
                y_ave.setEntry(i,new Mean().evaluate(o.getTrain().getRow(i)));
                for (int j = off_set; j<temp.getColumnDimension();j++){
                    if(Double.isNaN(test.getEntry(i,j))){
                        temp.setEntry(i,j,Double.NaN);
                    }
                    else if(Double.isNaN(forecast.getEntry(i,j))){
                        temp.setEntry(i,j,Double.MAX_VALUE);
                    }
                    else{
                        temp.setEntry(i,j,Math.pow(test.getEntry(i,j)-forecast.getEntry(i,j),2));
                    }

                }
            }
            int count = 0;
            for(int i = 0; i<temp.getRowDimension();i++){
                for (int j = off_set; j<temp.getColumnDimension();j++) {
                    if(Double.isNaN(temp.getEntry(i,j))){
                        continue;
                    }
                    rowAve.setEntry(i,rowAve.getEntry(i)+(temp.getEntry(i,j)-rowAve.getEntry(i))/++count);
                }
                count=0;
            }
            rowAve = MatrixUtils.createRealVector(ArrayUtils.pow(rowAve.toArray(),0.5));
            rowAve = rowAve.ebeDivide(y_ave);
            if(Double.isNaN(rowAve.getEntry(0))|| Double.isInfinite(rowAve.getEntry(0)))
                System.out.println(y_ave);
            return rowAve;

        }

        @Override
        public String getDescription() {
            return "ANRMSE";
        }
    }
    public static class MWUAS extends Metric{

        @Override
        public RealVector error_truncated_vec(TSModel.Output o, int off_set) {
            RealVector esti = o.error_vec(new MWUES());
            RealMatrix test = o.getTest();
            RealMatrix forecast = o.getForecast();
            RealVector ret = MatrixUtils.createRealVector(new double[test.getRowDimension()]);
            int count = 0;
            double tot = 0;
            for(int i = 0; i<test.getRowDimension();i++){
                if(esti.getEntry(i) == 0) {
                    ret.setEntry(i,0);
                    continue;
                }
                for(int j = off_set;j<test.getColumnDimension();j++){
                    if(test.getEntry(i,j)==0){
                        throw new Error("0 values are not allowed in the test set");
                    }
                    tot += test.getEntry(i,j)>forecast.getEntry(i,j) ? (test.getEntry(i,j) - forecast.getEntry(i,j))/Math.abs(test.getEntry(i,j)) : 0;
                    count++;
                }
                if(count!=0)
                    ret.setEntry(i,tot/(count*esti.getEntry(i)));
                count = 0;
                tot = 0;
            }
            return ret;
        }

        @Override
        public String getDescription() {
            return "MWUAS";
        }
    }
    public static class SMWUAS extends Metric{

        @Override
        public RealVector error_truncated_vec(TSModel.Output o, int off_set) {
            RealVector esti = o.error_vec(new MWUES());
            RealMatrix test = o.getTest();
            RealMatrix forecast = o.getForecast();
            RealVector ret = MatrixUtils.createRealVector(new double[test.getRowDimension()]);
            int count = 0;
            double tot = 0;
            for(int i = 0; i<test.getRowDimension();i++){
                if(esti.getEntry(i) == 0) {
                    ret.setEntry(i,0);
                    continue;
                }
                for(int j = off_set;j<test.getColumnDimension();j++){
                    if(test.getEntry(i,j)==0&&forecast.getEntry(i,j)==0){
                        ret.setEntry(i,Double.NaN);
                        break;
                    }
                    tot += test.getEntry(i,j)>forecast.getEntry(i,j) ? Math.abs(test.getEntry(i,j) - forecast.getEntry(i,j))/((Math.abs(test.getEntry(i,j))+Math.abs(forecast.getEntry(i,j)))/2) : 0;
                    count++;
                }
                if(count!=0)
                    ret.setEntry(i,tot/(count*esti.getEntry(i)));
                count = 0;
                tot = 0;
            }
            return ret;
        }

        @Override
        public String getDescription() {
            return "SMWUAS";
        }
    }
    public static class RMSE extends Metric{

        public RMSE() {

        }
        public String toString(){
            return "RMSE";
        }


        @Override
        public RealVector error_truncated_vec(TSModel.Output o, int off_set) {
            RealMatrix test = o.getTest();
            RealMatrix forecast = o.getForecast();
            RealMatrix temp = MatrixUtils.createRealMatrix(forecast.getRowDimension(),forecast.getColumnDimension());
            RealVector rowAve = MatrixUtils.createRealVector(new double[temp.getRowDimension()]);
            for(int i = 0; i<temp.getRowDimension();i++){
                for (int j = off_set; j<temp.getColumnDimension();j++){
                    if(Double.isNaN(test.getEntry(i,j))){
                        temp.setEntry(i,j,Double.NaN);
                    }
                    else if(Double.isNaN(forecast.getEntry(i,j))){
                        temp.setEntry(i,j,Double.MAX_VALUE);
                    }
                    else{
                        temp.setEntry(i,j,Math.pow(test.getEntry(i,j)-forecast.getEntry(i,j),2));
                    }

                }
            }
            int count = 0;
            for(int i = 0; i<temp.getRowDimension();i++){
                for (int j = off_set; j<temp.getColumnDimension();j++) {
                    if(Double.isNaN(temp.getEntry(i,j))){
                        continue;
                    }
                    rowAve.setEntry(i,rowAve.getEntry(i)+(temp.getEntry(i,j)-rowAve.getEntry(i))/++count);
                }
                count=0;
            }
            rowAve = MatrixUtils.createRealVector(ArrayUtils.pow(rowAve.toArray(),0.5));
            return rowAve;

        }

        @Override
        public String getDescription() {
            return "RMSE";
        }
    }
    public static class SRMSE extends Metric{
        //normalize the data set then compute RMSE

        public SRMSE() {

        }
        public String toString(){
            return "SRMSE";
        }


        @Override
        public RealVector error_truncated_vec(TSModel.Output o, int off_set) {
            RealMatrix test = MatrixTools.normalize(o.getTest());
            RealMatrix forecast = MatrixTools.normalize(o.getForecast());
            RealMatrix temp = MatrixUtils.createRealMatrix(forecast.getRowDimension(),forecast.getColumnDimension());
            RealVector rowAve = MatrixUtils.createRealVector(new double[temp.getRowDimension()]);
            for(int i = 0; i<temp.getRowDimension();i++){
                for (int j = off_set; j<temp.getColumnDimension();j++){
                    if(Double.isNaN(test.getEntry(i,j))){
                        temp.setEntry(i,j,Double.NaN);
                    }
                    else if(Double.isNaN(forecast.getEntry(i,j))){
                        temp.setEntry(i,j,Double.MAX_VALUE);
                    }
                    else{
                        temp.setEntry(i,j,Math.pow(test.getEntry(i,j)-forecast.getEntry(i,j),2));
                    }

                }
            }
            int count = 0;
            for(int i = 0; i<temp.getRowDimension();i++){
                for (int j = off_set; j<temp.getColumnDimension();j++) {
                    if(Double.isNaN(temp.getEntry(i,j))){
                        continue;
                    }
                    rowAve.setEntry(i,rowAve.getEntry(i)+(temp.getEntry(i,j)-rowAve.getEntry(i))/++count);
                }
                count=0;
            }
            rowAve = MatrixUtils.createRealVector(ArrayUtils.pow(rowAve.toArray(),0.5));
            return rowAve;

        }

        @Override
        public String getDescription() {
            return "SRMSE";
        }
    }
    public static class MRMSE extends Metric{
        //RMSE normalized by level

        public MRMSE() {

        }
        public String toString(){
            return "MRMSE";
        }


        @Override
        public RealVector error_truncated_vec(TSModel.Output o, int off_set) {
            RealMatrix test = o.getTest();
            RealMatrix forecast = o.getForecast();
            RealMatrix temp = MatrixUtils.createRealMatrix(forecast.getRowDimension(),forecast.getColumnDimension());
            RealVector rowAve = MatrixUtils.createRealVector(new double[temp.getRowDimension()]);
            RealVector y_ave = MatrixUtils.createRealVector(new double[temp.getRowDimension()]);
            for(int i = 0; i<temp.getRowDimension();i++){
                y_ave.setEntry(i,new Mean().evaluate(o.getTest().getRow(i)));
                for (int j = off_set; j<temp.getColumnDimension();j++){
                    if(Double.isNaN(test.getEntry(i,j))){
                        temp.setEntry(i,j,Double.NaN);
                    }
                    else if(Double.isNaN(forecast.getEntry(i,j))){
                        temp.setEntry(i,j,Double.MAX_VALUE);
                    }
                    else{
                        temp.setEntry(i,j,Math.pow(test.getEntry(i,j)-forecast.getEntry(i,j),2));
                    }

                }
            }
            int count = 0;
            for(int i = 0; i<temp.getRowDimension();i++){
                for (int j = off_set; j<temp.getColumnDimension();j++) {
                    if(Double.isNaN(temp.getEntry(i,j))){
                        continue;
                    }
                    rowAve.setEntry(i,rowAve.getEntry(i)+(temp.getEntry(i,j)-rowAve.getEntry(i))/++count);
                }
                count=0;
            }
            rowAve = MatrixUtils.createRealVector(ArrayUtils.pow(rowAve.toArray(),0.5));
            rowAve = rowAve.ebeDivide(y_ave);
            return rowAve;

        }

        @Override
        public String getDescription() {
            return "MRMSE";
        }
    }
    public static class NRMSE extends Metric{
        //RMSE normalized by difference min max

        public NRMSE() {

        }
        public String toString(){
            return "NRMSE";
        }


        @Override
        public RealVector error_truncated_vec(TSModel.Output o, int off_set) {
            RealMatrix test = o.getTest();
            RealMatrix forecast = o.getForecast();
            RealMatrix temp = MatrixUtils.createRealMatrix(forecast.getRowDimension(),forecast.getColumnDimension());
            RealVector rowAve = MatrixUtils.createRealVector(new double[temp.getRowDimension()]);
            RealVector y_ave = MatrixUtils.createRealVector(new double[temp.getRowDimension()]);
            for(int i = 0; i<temp.getRowDimension();i++){
                double min = Collections.min(Arrays.asList(ArrayUtils.toObject(o.getTest().getRow(i))));
                double max = Collections.max(Arrays.asList(ArrayUtils.toObject(o.getTest().getRow(i))));
                y_ave.setEntry(i,max-min);
                for (int j = off_set; j<temp.getColumnDimension();j++){
                    if(Double.isNaN(test.getEntry(i,j))){
                        temp.setEntry(i,j,Double.NaN);
                    }
                    else if(Double.isNaN(forecast.getEntry(i,j))){
                        temp.setEntry(i,j,Double.MAX_VALUE);
                    }
                    else{
                        temp.setEntry(i,j,Math.pow(test.getEntry(i,j)-forecast.getEntry(i,j),2));
                    }

                }
            }
            int count = 0;
            for(int i = 0; i<temp.getRowDimension();i++){
                for (int j = off_set; j<temp.getColumnDimension();j++) {
                    if(Double.isNaN(temp.getEntry(i,j))){
                        continue;
                    }
                    rowAve.setEntry(i,rowAve.getEntry(i)+(temp.getEntry(i,j)-rowAve.getEntry(i))/++count);
                }
                count=0;
            }
            rowAve = MatrixUtils.createRealVector(ArrayUtils.pow(rowAve.toArray(),0.5));
            rowAve = rowAve.ebeDivide(y_ave);
            return rowAve;

        }

        @Override
        public String getDescription() {
            return "NRMSE";
        }
    }

    public static class MASE extends Metric{
        private HashMap<Integer,RealVector> map = new HashMap<>();
        private boolean buffer;
        public MASE(Collapse c){
            super.collapse = c;
        }
        public MASE(){

        }
        public static Metric createMetric(String s) {
            if(s.equalsIgnoreCase("RMSE"))
                return new RMSE();
            else if(s.equalsIgnoreCase("MAE"))
                return new MAE();
            else
                throw new IllegalArgumentException("thats not a supported metric");
        }
        @Override
        public RealVector error_truncated_vec(TSModel.Output o, int off_set) {
            RealMatrix train = o.getTrain();
            RealMatrix test = o.getTest();
            RealMatrix forecast = o.getForecast();
            TSModel.Output mae_in = TSModel.dummy().new Output();
            RealVector tot = MatrixUtils.createRealVector(ArrayUtils.zeros(test.getRowDimension()));
            double count = 0;
            RealVector base;
            Metric mae = new MAE();
            if(buffer&&map.containsKey(test.getColumnDimension())){
                base = tot.add(map.get(test.getColumnDimension()));
            }
            else {
                for (int i = 1; i < train.getColumnDimension() - test.getColumnDimension() - 1; i++) {


                    //compute MAE of random walk model with horizon of test length
                    mae_in.setTest(train.getSubMatrix(0, train.getRowDimension() - 1, i, i + test.getColumnDimension() - 1));
                    mae_in.setForecast(MatrixTools.repeatCollumnVector(train.getColumnVector(i - 1), test.getColumnDimension()));
                    RealVector mae_error = mae_in.error_truncated_vec(mae,off_set);
                    tot = tot.add(mae_error);
                    count++;
                }

                base = tot.mapDivide(count);
            }
            if(buffer) {
                map.put(test.getColumnDimension(), base);
            }
            mae_in.setForecast(forecast);
            mae_in.setTest(test);
            return mae_in.error_truncated_vec(mae,off_set).ebeDivide(base);


        }

        @Override
        public String getDescription() {
            return "MASE";
        }

        public void reset_buffer() {
            map = new HashMap<>();
        }
    }

    public static class SMAPE extends Metric{

        @Override
        public RealVector error_truncated_vec(TSModel.Output o, int off_set) {
            RealVector r = MatrixUtils.createRealVector(new double[o.getTest().getColumnDimension()-off_set]);
            RealVector error = MatrixUtils.createRealVector(new double[o.getForecast().getRowDimension()]);
            for(int i = 0; i<o.getTest().getRowDimension();i++){
                for(int j = off_set; j<o.getForecast().getColumnDimension();j++){
                    r.setEntry(j-off_set,Math.abs(o.getTest().getEntry(i,j)-o.getForecast().getEntry(i,j))/((Math.abs(o.getTest().getEntry(i,j))+Math.abs(o.getForecast().getEntry(i,j)))/2));
                }
                error.setEntry(i,new Mean().evaluate(r.toArray()));

            }
            return error;
        }

        @Override
        public String getDescription() {
            return "SMAPE";
        }
    }

    public static class MAE extends Metric {
        @Override
        public RealVector error_truncated_vec(TSModel.Output o, int off_set) {
            RealMatrix test = o.getTest();
            RealMatrix forecast = o.getForecast();
            RealVector sum = MatrixUtils.createRealVector(ArrayUtils.zeros(test.getRowDimension()));

            double dif;
            int count1 = 0;
            for(int i = 0; i<forecast.getRowDimension();i++){
                for(int j = off_set; j<forecast.getColumnDimension();j++){
                    if(!Double.isNaN(test.getEntry(i,j))){
                        dif = Math.abs(forecast.getEntry(i,j)-test.getEntry(i,j));
                        sum.setEntry(i,sum.getEntry(i)+(dif-sum.getEntry(i))/++count1);
                    }
                }
                count1 = 0;
            }
            return sum;

        }

        @Override
        public String getDescription() {
            return "MAE";
        }
    }
}