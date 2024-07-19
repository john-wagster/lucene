package org.apache.lucene.sandbox.rabitq;

public class Utils {
  public static double sqrtNewtonRaphson(double x, double curr, double prev) {
    return (curr == prev) ? curr : sqrtNewtonRaphson(x, 0.5 * (curr + x / curr), curr);
  }

  public static double constSqrt(double x) {
    return x >= 0 && !Double.isInfinite(x) ? sqrtNewtonRaphson(x, x, 0) : Double.NaN;
  }
}
