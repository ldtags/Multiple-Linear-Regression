/*
 * Author: Liam Tangney
 */

import java.util.ArrayList;
import java.util.List;

public class DataPoint {
    public List<Double> x;
    public double y;

    public DataPoint(List<Double> x, double y) {
        this.x = x;
        this.y = y;
    }

    public List<Double> getX() {
        return this.x;
    }

    public double getY() {
        return this.y;
    }

    public void augment(int degree) {
        List<Double> augmentedData = new ArrayList<>();
        augmentedData.add(1.0);
        augmentedData.addAll(this.x);
        for (int i = 2; i < degree + 1; i++) {
            for (double val : this.x) {
                augmentedData.add(Math.pow(val, i));
            }
        }

        this.x = augmentedData;
    }
}
