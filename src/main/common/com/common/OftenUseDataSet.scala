package com.common

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

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