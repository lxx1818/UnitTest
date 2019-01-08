package com.common

import org.apache.spark.ml.attribute._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

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

object OftenUseDataSet {

  private val classifierPath = "/home/lxx/dataset/classification/"

  final val featureColName = "featureVe"
  final val labelColName = "label"

  def getIrisDataSet(sparkSession: SparkSession): DataFrame = {
    getDataSet(5, 4, classifierPath + "iris.data", sparkSession)
  }

  def getWineDataSet(sparkSession: SparkSession): DataFrame = {
    getDataSet(14, 0, classifierPath + "wine.data", sparkSession)
  }

  def getHaberman(sparkSession: SparkSession): DataFrame = {
    getDataSet(4, 3, classifierPath + "haberman.data", sparkSession)
  }

  def getGerman(sparkSession: SparkSession): DataFrame = {
    getDataSet(25, 24, classifierPath + "german.data-numeric", sparkSession)
  }

  def getDataSet(
      numCol: Int,
      labelIndex: Int,
      filePath: String,
      sparkSession: SparkSession): DataFrame = {

    val dataset = sparkSession.read.format("csv")
      .option("header", "false")
      .option("inferSchema", true.toString)
      .load(filePath)

    val colName = Tools.getColName(numCol)
    val label = colName(labelIndex)

    val resultDF = new VectorAssembler()
      .setInputCols(colName.filterNot(_.equals(label)))
      .setOutputCol(featureColName)
      .transform(dataset)

    val indexDF = new StringIndexer()
      .setInputCol(label).setOutputCol(labelColName)
      .fit(resultDF).transform(resultDF)
    indexDF.drop(colName: _*)

  }

}