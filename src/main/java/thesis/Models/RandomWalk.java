package thesis.Models;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import thesis.TSModel;

import java.util.ArrayList;
import java.util.HashMap;

public class RandomWalk extends TSModel {
    @Override
    protected void parse(HashMap<String, ArrayList<String>> in) {

    }

    @Override
    protected Output evaluate() {
        RealVector last = o.getTrain().getColumnVector(o.getTrain().getColumnDimension()-1);
        RealMatrix forecast = MatrixUtils.createRealMatrix(o.getTest().getRowDimension(),o.getTest().getColumnDimension());
        for(int i = 0; i<forecast.getColumnDimension();i++) {
            forecast.setColumnVector(i,last);
        }
        o.setForecast(forecast);
        return o;
    }

    @Override
    public boolean is1D() {
        return true;
    }

    @Override
    public boolean isND() {
        return true;
    }

    @Override
    public boolean legal_hyperparameters(RealMatrix train) {
        return true;
    }

    @Override
    public boolean missingValuesAllowed() {
        return false;
    }

    @Override
    public HashMap<String, String[]> getSearchSpace() {
        return null;
    }
}
