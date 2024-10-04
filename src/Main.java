/*
 * Author: Liam Tangney
 */

import java.io.File;

public class Main {
    public static void main(String[] args) throws Exception {
        String filename = null;
        Integer folds = 1;
        Integer minPolynomial = 1;
        Integer maxPolynomial = null;
        Double learningRate = 0.005;
        Integer epochLimit = 10000;
        Integer batchSize = 0;
        Boolean randomization = false;
        Integer verbosity = 1;
        Integer i = 0;

        while (i < args.length) {
            try {
                switch (args[i]) {
                case "-f":
                    filename = args[++i];
                    File f = new File(filename);
                    if (!f.exists()) {
                        System.err.println("No file named " + filename + " exists in the working directory");
                        return;
                    }

                    if (f.isDirectory()) {
                        System.err.println("The file named " + filename + " must be a text file, not a directory");
                        return;
                    }
                    break;
                case "-k":
                    try {
                        folds = Integer.parseInt(args[++i]);
                    } catch (NumberFormatException err) {
                        System.err.println("-k must be followed by an integer");
                        return;
                    }
                    break;
                case "-d":
                    try {
                        minPolynomial = Integer.parseInt(args[++i]);
                    } catch (NumberFormatException err) {
                        System.err.println("-d must be followed by an integer");
                        return;
                    }
                    break;
                case "-D":
                    try {
                        maxPolynomial = Integer.parseInt(args[++i]);
                    } catch (NumberFormatException err) {
                        System.err.println("-D must be followed by an integer");
                        return;
                    }
                    break;
                case "-a":
                    try {
                        learningRate = Double.parseDouble(args[++i]);
                    } catch (NumberFormatException err) {
                        System.err.println("-a must be followed by a double");
                        return;
                    }
                    break;
                case "-e":
                    try {
                        epochLimit = Integer.parseInt(args[++i]);
                    } catch (NumberFormatException err) {
                        System.err.println("-e must be followed by an integer");
                        return;
                    }
                    break;
                case "-m":
                    try {
                        batchSize = Integer.parseInt(args[++i]);
                    } catch (NumberFormatException err) {
                        System.err.println("-m must be followed by an integer");
                        return;
                    }
                    break;
                case "-r":
                    randomization = true;
                    break;
                case "-v":
                    try {
                        verbosity = Integer.parseInt(args[++i]);
                    } catch (NumberFormatException err) {
                        System.err.println("-v must be followed by an integer");
                        return;
                    }

                    if (verbosity < 1 || verbosity > 5) {
                        System.err.println(verbosity + " is not a valid verbosity level");
                        System.err.println("Valid verbosity levels: [1 | 2 | 3 | 4 | 5]");
                        return;
                    }
                }
            } catch (IndexOutOfBoundsException err) {
                if (i >= args.length) {
                    System.err.println("An unexpected error occurred");
                    return;
                }

                System.err.println(args[i] + " must be followed by a value");
                return;
            }
            i++;
        }

        if (maxPolynomial == null) {
            maxPolynomial = minPolynomial;
        } else if (maxPolynomial < minPolynomial) {
            System.err.println("Max polynomial degree cannot be less than the min polynomial degree");
            return;
        }

        Agent agent = new Agent(learningRate, epochLimit, batchSize, randomization, verbosity);
        agent.loadData(filename);
        agent.start(minPolynomial, maxPolynomial, folds);
    }
}
