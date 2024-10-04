/*
 * Author: Liam Tangney
 */

import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.BufferedReader;

public class Agent {
    private static final double DELTA_COST_LIMIT = Math.pow(10, -10);

    private List<DataPoint> data = null;
    private double learningRate = 0.005;
    private int epochLimit = 10000;
    private int batchSize = 0;
    private boolean randomize = false;
    private int verbosity = 1;

    public Agent(double learningRate, int epochLimit, int batchSize, boolean randomize, int verbosity) {
        this.data = new ArrayList<>();
        this.learningRate = learningRate;
        this.epochLimit = epochLimit;
        this.batchSize = batchSize;
        this.randomize = randomize;
        this.verbosity = verbosity;
    }

    public void setLearningRate(Double learningRate) {
        this.learningRate = learningRate;
    }

    public double getLearningRate() {
        return this.learningRate;
    }

    public void setEpochLimit(Integer epochLimit) {
        this.epochLimit = epochLimit;
    }

    public int getEpochLimit() {
        return this.epochLimit;
    }

    public void setBatchSize(Integer batchSize) {
        this.batchSize = batchSize;
    }

    public int getBatchSize() {
        return this.batchSize;
    }

    public void setVerbosity(Integer verbosity) {
        this.verbosity = verbosity;
    }

    public int getVerbosity() {
        return this.verbosity;
    }

    public void setRandomization(Boolean randomize) {
        this.randomize = randomize;
    }

    public boolean getRandomization() {
        return this.randomize;
    }

    public void loadData(String filename) throws IOException {
        File file = new File(filename);
        if (!file.exists()) {
            throw new IOException("No file named " + filename + " exists");
        }

        if (file.isDirectory()) {
            throw new IOException("Data file cannot be a directory");
        }

        List<DataPoint> dataStore = new ArrayList<>();
        String line;
        String[] splitLine;
        int lineNum = 1;
        BufferedReader reader = new BufferedReader(new FileReader(file));
        try {
            while ((line = reader.readLine()) != null) {
                if (line.charAt(0) != '#') {
                    splitLine = line.split(" ");
                    if (splitLine.length < 2) {
                        throw new IOException("Illegal data format at line " + lineNum);
                    }

                    List<Double> x = new ArrayList<>();
                    for (int i = 0; i < splitLine.length - 1; i++) {
                        try {
                            x.add(Double.parseDouble(splitLine[i]));
                        } catch (NumberFormatException err) {
                            throw new IOException("Invalid data format: " + splitLine[i]);
                        }
                    }

                    double y = Double.parseDouble(splitLine[splitLine.length - 1]);
                    dataStore.add(new DataPoint(x, y));
                }
                lineNum++;
            }
        } finally {
            reader.close();
        }

        this.data.clear();
        for (DataPoint point : dataStore) {
            this.data.add(point);
        }
    }

    public List<DataPoint> getData() {
        return this.data;
    }

    private List<List<DataPoint>> foldData(int k) {
        List<DataPoint> data = this.data;
        List<List<DataPoint>> folds = new ArrayList<>();

        if (this.randomize) {
            data = new ArrayList<>(data);
            Collections.shuffle(data);
        }

        for (int i = 0; i < data.size(); i++) {
            if (folds.size() <= i % k) {
                folds.add(new ArrayList<>());
            }

            folds.get(i % k).add(data.get(i));
        }

        return folds;
    }

    private static List<DataPoint> augmentData(List<DataPoint> data, int degree) {
        List<DataPoint> augmentedData = new ArrayList<>();
        DataPoint augmentedPoint = null;

        for (DataPoint point : data) {
            augmentedPoint = new DataPoint(point.x, point.y);
            augmentedPoint.augment(degree);
            augmentedData.add(augmentedPoint);
        }

        return augmentedData;
    }

    private static Double calculateDotProduct(List<Double> a, List<Double> b) {
        Double result = 0.0;

        for (int i = 0; i < a.size(); i++) {
            result += a.get(i) * b.get(i);
        }

        return result;
    }

    private static double calculateGradient(List<DataPoint> data, List<Double> weights, int k, int vectorSize) {
        double gradient;
        double sum;

        if (data.size() == 0) {
            return 0;
        }

        gradient = 0;
        for (int i = 0; i < data.size(); i++) {
            sum = 0;
            for (int j = 0; j < vectorSize; j++) {
                sum += weights.get(j) * data.get(i).getX().get(j);
            }

            gradient += -2.0 * data.get(i).getX().get(k) * (data.get(i).getY() - sum);
        }

        gradient *= 1.0 / (data.size() * 1.0);

        return gradient;
    }

    private static double calculateLoss(double expected, double actual) {
        return Math.pow(actual - expected, 2);
    }

    private static double calculateCost(List<DataPoint> data, List<Double> weights) {
        double sum = 0.0;
        double expected = 0.0;
        double actual = 0.0;

        for (int i = 0; i < data.size(); i++) {
            expected = data.get(i).getY();
            actual = calculateDotProduct(data.get(i).x, weights);
            sum += calculateLoss(expected, actual);
        }

        return sum / data.size();
    }

    private static String getFormattedModel(List<Double> weights, int degree, int padding) {
        String formattedModel = "";
        double weight;
        int degreeSize = (weights.size() - 1) / degree;
        int currentDegree;

        for (int i = 0; i < padding; i++) {
            formattedModel += " ";
        }

        formattedModel += String.format("Model: Y = %.4f", weights.get(0));
        for (int i = 1; i < weights.size(); i++) {
            weight = weights.get(i);
            if (weight < 0) {
                formattedModel += String.format(" - %.4f X%d", weight * -1, i);
            } else {
                formattedModel += String.format(" + %.4f X%d", weights.get(i), i);
            }

            currentDegree = (i - 1) / degreeSize + 1;
            if (currentDegree > 1) {
                formattedModel += String.format("^%d", currentDegree);
            }
        }

        return formattedModel;
    }

    private static void reportTrainingError(List<DataPoint> data, List<Double> weights) {
        double error;
        String formattedError = null;
        String reportString = "  * Training error";

        error = calculateCost(data, weights);
        formattedError = String.format("%.6f", error);
        reportString += String.format("%15s", formattedError);

        System.out.printf("%s\n\n", reportString);
    }

    private static void reportTrainingError(List<DataPoint> data, List<DataPoint> validationData,
            List<Double> weights) {
        double error;
        String formattedError = null;
        String reportString = "  * Training and validation errors";

        error = calculateCost(data, weights);
        formattedError = String.format("%.6f", error);
        reportString += String.format("%15s", formattedError);

        error = calculateCost(validationData, weights);
        formattedError = String.format("%.6f", error);
        reportString += String.format("%14s", formattedError);

        System.out.printf("%s\n\n", reportString);
    }

    private static void reportModel(List<Double> weights, int degree) {
        System.out.println(getFormattedModel(weights, degree, 6));
    }

    private void reportCost(List<DataPoint> data, List<Double> weights, int epochs, int iterations, int degree,
            int padding) {
        double cost = calculateCost(data, weights);
        String formattedCost = String.format("%.9f", cost);
        String paddingString = "";
        String preString = null;
        String reportString = null;

        for (int i = 0; i < padding; i++) {
            paddingString += " ";
        }

        if (epochs == 0) {
            preString = String.format("%sInitial model with zero weights   :", paddingString);
        } else {
            preString = String.format("%sAfter %6s epochs (%6s iter.):", paddingString, epochs, iterations);
        }

        reportString = String.format("%s Cost%15s", preString, formattedCost);
        if (this.verbosity >= 4) {
            reportString += getFormattedModel(weights, degree, 3);
        }

        System.out.println(reportString);
    }

    /**
     * Splits data into a collection of batches. Each batch will be as evenly sized
     * as
     * possible.
     * 
     * @param data      Training data being split into batches
     * @param batchSize The maximum size of each batch
     * 
     * @return The collection of batches
     */
    private List<List<DataPoint>> getBatches(List<DataPoint> data, int batchSize) {
        int batchCount, batchNum;
        List<DataPoint> randomData = null;
        List<List<DataPoint>> batches = null;

        if (batchSize <= 1) {
            batches = new ArrayList<>(1);
            batches.add(data);
            return batches;
        }

        batchCount = Math.ceilDiv(data.size(), batchSize);
        batches = new ArrayList<>(batchCount);
        if (this.randomize) {
            randomData = new ArrayList<>(data);
            Collections.shuffle(randomData);
            data = randomData;
        }

        for (int i = 0; i < data.size(); i++) {
            batchNum = Math.floorDiv(i, batchSize);
            if (batches.size() <= batchNum) {
                batches.add(new ArrayList<>());
            }

            batches.get(batchNum).add(data.get(i));
        }

        return batches;
    }

    /**
     * Uses mini-batch gradient descent to fit a model for multiple linear
     * regression.
     * 
     * @param data         Augmented training data
     * @param learningRate Learning rate for mini-batch gradient descent
     * @param batchSize    Size of the batches in mini-batch gradient descent
     * @param degree       The polynomial degree of the model being fit
     * 
     * @return The weights for the model that was fit with the augmented training
     *         data
     */
    private List<Double> fit(List<DataPoint> data, double learningRate, int batchSize, int degree) {
        short stopCondition = 0;
        int vectorSize = data.get(0).x.size();
        int t = 0;
        int e = 0;
        long startTime = System.currentTimeMillis();
        long timeElapsed;
        double gradient;
        double startingCost;
        double deltaCost;
        double newCost;
        List<Double> weightVector = new ArrayList<>(Collections.nCopies(vectorSize, 0.0));
        List<List<DataPoint>> batches = null;

        if (batchSize == 0) {
            batchSize = 1;
        }

        if (this.verbosity >= 2) {
            System.out.println("    * Beginning mini-batch gradient descent");
            System.out.printf("      (alpha=%.6f, epochLimit=%d, batchSize=%d)\n", learningRate, this.epochLimit,
                    batchSize);
        }

        if (this.verbosity >= 3) {
            reportCost(data, weightVector, e, t, degree, 6);
        }

        while (e <= this.epochLimit) {
            startingCost = calculateCost(data, weightVector);
            batches = getBatches(data, batchSize);
            for (List<DataPoint> batch : batches) {
                for (int k = 0; k < vectorSize; k++) {
                    gradient = calculateGradient(batch, weightVector, k, vectorSize);
                    weightVector.set(k, weightVector.get(k) - (learningRate * gradient));
                }

                t++;
            }

            e++;
            if (this.verbosity >= 3) {
                if (e % 1000 == 0 || this.verbosity >= 5) {
                    this.reportCost(data, weightVector, e, t, degree, 6);
                }
            }

            if (stopCondition == 1) {
                break;
            }

            newCost = calculateCost(data, weightVector);
            if (newCost < DELTA_COST_LIMIT) {
                stopCondition = 1;
            }

            deltaCost = Math.abs(newCost - startingCost);
            if (deltaCost < DELTA_COST_LIMIT) {
                stopCondition = 1;
            }
        }

        if (this.verbosity >= 3 && this.verbosity < 5) {
            this.reportCost(data, weightVector, e, t, degree, 6);
        }

        timeElapsed = System.currentTimeMillis() - startTime;
        if (this.verbosity >= 2) {
            System.out.println("    * Done with fitting!");
            System.out.printf("      Training took %dms, %d epochs, %d iterations", timeElapsed, e, t);
            System.out.printf(" (%.4fms / iteration)\n", ((timeElapsed * 1.0) / (t * 1.0)));
            String stopConditionString = "";
            switch (stopCondition) {
                case 0:
                    stopConditionString = "Epoch Limit";
                    break;
                case 1:
                    stopConditionString = "DeltaCost ~= 0";
                    break;
            }
            System.out.println("      GD Stop condition: " + stopConditionString);
            reportModel(weightVector, degree);
        }

        return weightVector;
    }

    /**
     * Performs multiple linear regression with mini-batch gradient descent to train
     * a model.
     * 
     * Training data must be loaded prior to the execution of this method.
     * 
     * @param minPolynomial The minimum degree of polynomial that the model will fit
     * @param maxPolynomial The maximum degree of polynomial that the model will fit
     * @param k             The amount of folds used in K-Fold Cross Validation (if
     *                      1, K-Fold Cross Validation will be skipped)
     */
    public void start(Integer minPolynomial, Integer maxPolynomial, Integer k) {
        List<Double> weights = null;
        List<DataPoint> data = null;
        List<List<DataPoint>> folds = null;

        if (this.data == null) {
            System.err.println("No training data detected.");
            System.err.println("Please load training data into the Agent with the loadData method");
            return;
        }

        if (k > 1) {
            folds = this.foldData(k);
            System.out.println("Using " + k + "-fold cross-validation.");
        } else {
            System.out.println("Skipping cross-validation.");
        }

        for (int d = minPolynomial; d < maxPolynomial + 1; d++) {
            System.out.println("----------------------------------");
            System.out.println("* Using a model of degree " + d);

            if (k > 1) {
                for (int i = 0; i < folds.size(); i++) {
                    System.out.println("  * Training on all data except Fold " + (i + 1) + " ("
                            + (this.data.size() - folds.get(i).size()) + " examples):");
                    data = new ArrayList<>();
                    for (int j = 0; j < folds.size(); j++) {
                        if (j != i) {
                            data.addAll(folds.get(j));
                        }
                    }

                    data = augmentData(data, d);
                    weights = this.fit(data, this.learningRate, this.batchSize, d);
                    reportTrainingError(data, augmentData(folds.get(i), d), weights);
                }
            } else {
                System.out.println("  * Training on all data (" + this.data.size() + " examples):");
                data = augmentData(this.data, d);
                weights = this.fit(data, this.learningRate, this.batchSize, d);
                reportTrainingError(data, weights);
            }
        }
    }
}
