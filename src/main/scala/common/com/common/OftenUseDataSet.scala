package com.common

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

object OftenUseDataSet {

  private val classifierPath = "./dataset/classification/"
//  private val classifierPath = "/lxxDataSet/classification/"

  final val featureColName = "featureVe"
  final val labelColName = "label"

  // iris数据集共有3个label
  def getIrisDataSet(sparkSession: SparkSession): DataFrame = {
    getDataSet(5, 4, classifierPath + "iris.data", sparkSession)
  }

  // wine数据集共有3个label
  def getWineDataSet(sparkSession: SparkSession): DataFrame = {
    getDataSet(14, 0, classifierPath + "wine.data", sparkSession)
  }

  // Haberman数据集共有2个label
  def getHaberman(sparkSession: SparkSession): DataFrame = {
    getDataSet(4, 3, classifierPath + "haberman.data", sparkSession)
  }

  // German Credit Data数据集共有2个label
  def getGerman(sparkSession: SparkSession): DataFrame = {
    getDataSet(25, 24, classifierPath + "german.data-numeric", sparkSession)
  }

  def getOnlineRetail(sparkSession: SparkSession): DataFrame = {
    var indexDF = sparkSession.read.format("csv")
      .option("header", "true")
      .option("inferSchema", true.toString)
      .load("./dataset/classification/OnlineRetailOK.csv")
//      .load("/lxxDataSet/FMFeaturesDS/OnlineRetailOK.csv")

    indexDF = new StringIndexer()
      .setInputCol("StockCode").setOutputCol("DescriptionIndex")
      .fit(indexDF).transform(indexDF).drop("StockCode")
    indexDF = new StringIndexer()
      .setInputCol("Country").setOutputCol("CountryIndex")
      .fit(indexDF).transform(indexDF).drop("Country")

    // 过滤空值
    val names = indexDF.schema.map(_.name)
    indexDF.na.drop(names)
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