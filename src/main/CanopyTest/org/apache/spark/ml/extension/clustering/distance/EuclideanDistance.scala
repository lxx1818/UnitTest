package org.apache.spark.mllib

import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils

/**
  * @author XiangxiangLi
  *
  */
object EuclideanDistance {

  def getfastSquaredDistance(v1: MLVector, norm1: Double, v2: MLVector, norm2: Double): Double = {
    MLUtils.fastSquaredDistance(Vectors.fromML(v1), norm1, Vectors.fromML(v2), norm2)
  }
}
