package com.common

import java.io.PrintWriter

import org.apache.spark.graph.embedding.AliasSampler
import org.apache.spark.ml.attribute._
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object Tools {

  /**
    * 返回默认的列名
    * num: 列数
    * n: 起始位置
    */
  def getColName(num: Int, n: Int = 0): Array[String] = {
    val colName = Array.fill(num)("")
      for (i <- 0 until num) {
        colName(i) = "_c" + (i + n).toString
      }
    colName
  }

  def getSparkSession(): SparkSession = {
    SparkSession
      .builder()
      .master("local[*]")
      .appName("LXXSparkSession")
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

  def getAccuracyBC(dataset: Dataset[_]): Double = {
    new BinaryClassificationEvaluator()
      .setLabelCol(OftenUseDataSet.labelColName)
      .setMetricName("areaUnderROC")
      .evaluate(dataset)
  }

  def getNumClasses(dataset: Dataset[_]): Int = {
    GetNumClasses.getNumClasses(dataset.schema(OftenUseDataSet.labelColName)) match {
      case Some(n: Int) => n
      case None => dataset.select(OftenUseDataSet.labelColName).distinct().count().toInt
    }
  }

  // 10折交叉划分数据集
  def kFold(
      dataFrame: DataFrame,
      numFolds: Int = 10,
      seed: Int = 2): Array[(DataFrame, DataFrame)] = {
    val context = dataFrame.sqlContext
    val schema = dataFrame.schema
    MLUtils.kFold(dataFrame.rdd, numFolds, seed).map { case (training, validate) =>
      val trainDF = context.createDataFrame(training, schema)
      val testDF = context.createDataFrame(validate, schema)
      (trainDF, testDF)
    }
  }

  /**
   * 生成数据集
   * @param rowNum 数据集行数
   * @param featureNum 数据集特征列数（不包含label列）
   * @param label 是否含有label列，如果有，将会在最后一列
   * @param featureLableCount 特征列class个数(空值不被当作一个class)
   * @param lableCount label列class个数
   * @param nullRatio 空值比例（特征列中）
   * @param spark SparkSession
   * @return 生成数据集文件名
   */

  def generateDataSet(
      rowNum: Int,
      featureNum: Int,
      partitionNum: Int,
      spark: SparkSession,
      label: Boolean = false,
      featureLableCount: Int = 10,
      lableCount: Int = 2,
      nullRatio: Double = 0,
      path: String = "./"): String = {
    // 生成文件名
    val fileName = s"rowNum${rowNum}_featureNum${featureNum}_label${label}_nullRatio${nullRatio}"

    // 生成特征列标签可能性比例
    val sampleList = if (nullRatio == 0.0) {
      Array.fill(featureLableCount)(1.0 / featureLableCount)
    } else {
      Array.fill(featureLableCount)((1.0 - nullRatio) / featureLableCount) ++ Array(nullRatio)
    }

    // 产生随机数生成器
    val sampler = new AliasSampler(sampleList)

    val rddFile = spark.sparkContext.parallelize(1 to rowNum)
      .repartition(partitionNum).map { i =>
      // 产生空值列
      val samples = if (nullRatio == 0.0) {
        sampler.sample(featureNum).mkString(",")
      } else {
        sampler.sample(featureNum).map{ feature =>
          if (feature == featureLableCount) "" else feature.toString
        }.mkString(",")
      }
      // 判断是否含有label列
      if (label) {
        val label = scala.util.Random.nextInt(lableCount).toString
        samples + "," + label
      } else {
        samples
      }
    }

    // 写入数据集
    rddFile.saveAsTextFile(path + fileName)
    fileName
  }

  def readDataFrameASInteger(path: String, spark: SparkSession): DataFrame = {
    val rdd = spark.sparkContext.textFile(path).map(_.split(",", -1))
    val colsNum = rdd.first().length

    val rddRow = rdd.map { arr =>
      var row = Row()
      for (i <- 0 until colsNum) {
        row = Row.merge(row, Row(arr(i)))
      }
      row
    }

    val colNames = Array.fill(colsNum)("")
    var structFieldArr = Array[StructField]()
    for (i <- 0 until colsNum) {
      colNames(i) = "_c" + i
      structFieldArr = structFieldArr ++ Array(StructField(colNames(i),StringType,true))
    }

    val structType = StructType(structFieldArr)
    val df = spark.sqlContext.createDataFrame(rddRow,structType)
    df.withColumns(colNames, colNames.map(df.col(_).cast(IntegerType)))
  }

  // 写入文件
  def write(path: String, text: String): Unit = {
    val out = new PrintWriter(path)
    out.println(text)
    out.close
  }

  // 写入文件
  def write(path: String, textArr: Array[String]): Unit = {
    write(path, textArr.mkString("\n"))
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

