package org.apache.spark.ml.extension.clustering.distance

import org.apache.spark.ml.linalg._
import org.apache.spark.mllib.EuclideanDistance

/**
  * Like EuclideanDistanceMeasure but it does not take the square root.
  * Thus, it is not actually the Euclidean Distance, but it is saves on computation when you
  * only need the distance for comparison and don't care about the actual value as a distance.
  */
class SquaredEuclideanDistanceMeasure extends DistanceMeasure {

  /**
    * @return the distance between two points.
    */
  override def distance(v1: Vector, v2: Vector): Double = {
    Vectors.sqdist(v1, v2)
  }

  /**
    * If you have special skills, use these skills to calculate the distance.
    *
    * @return the squared Euclidean distance between two vectors computed by
    * [[org.apache.spark.mllib.util.MLUtils#fastSquaredDistance]].
    */
  override def fastDistance(v1: VectorWithNorm, v2: VectorWithNorm): Double = {
    EuclideanDistance.getfastSquaredDistance(v1.vector, v1.norm, v2.vector, v2.norm)
  }

  /**
    * @return the index of the closest center to the given point, as well as the squared distance.
    */
  override def findClosest(
      centers: TraversableOnce[VectorWithNorm],
      point: VectorWithNorm): (Int, Double) = {
    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0
    centers.foreach { center =>
      // Since `\|a - b\| \geq |\|a\| - \|b\||`, we can use this lower bound to avoid unnecessary
      // distance computation.
      var lowerBoundOfSqDist = center.norm - point.norm
      lowerBoundOfSqDist = lowerBoundOfSqDist * lowerBoundOfSqDist
      if (lowerBoundOfSqDist < bestDistance) {
        val distance = EuclideanDistance
          .getfastSquaredDistance(center.vector, center.norm, point.vector, point.norm)
        if (distance < bestDistance) {
          bestDistance = distance
          bestIndex = i
        }
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }

}
