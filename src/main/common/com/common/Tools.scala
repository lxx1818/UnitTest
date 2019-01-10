package com.common

import org.apache.spark.ml.attribute._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.{Dataset, SparkSession}

object Tools {

  /**
    * 返回默认的列名
    */
  def getColName(num: Int): Array[String] = {
    val colName = Array.fill(num)("")
      for (i <- 0 until num) {
        colName(i) = "_c" + i.toString
      }
    colName
  }

  def getSparkSession(): SparkSession = {
    SparkSession
      .builder()
      .master("local[*]")
      .appName("SparkSession")
      .getOrCreate()
  }

  /**
    * 多分类情况下，预测精度
    * @param dataset
    * @return
    */
  def genAccuracyMC(dataset: Dataset[_]): Double = {
    new MulticlassClassificationEvaluator()
      .setLabelCol(OftenUseDataSet.labelColName)
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
      .evaluate(dataset)
  }

  def getNumClasses(dataset: Dataset[_]): Int = {
    GetNumClasses.getNumClasses(dataset.schema(OftenUseDataSet.labelColName)) match {
      case Some(n: Int) => n
      case None => dataset.select(OftenUseDataSet.labelColName).distinct().count().toInt
    }
  }

}

object GetNumClasses {
  /**
    * Examine a schema to identify the number of classes in a label column.
    * Returns None if the number of labels is not specified, or if the label column is continuous.
    */
  def getNumClasses(labelSchema: StructField): Option[Int] = {
    Attribute.fromStructField(labelSchema) match {
      case binAttr: BinaryAttribute => Some(2)
      case nomAttr: NominalAttribute => nomAttr.getNumValues
      case _: NumericAttribute | UnresolvedAttribute => None
    }
  }
}

