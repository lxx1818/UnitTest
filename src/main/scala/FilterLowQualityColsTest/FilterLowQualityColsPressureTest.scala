package FilterLowQualityColsTest

import java.io.PrintWriter

import com.common.Tools
import org.apache.spark.ml.preprocess.FilterLowQualityCols
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.io.Source


object FilterLowQualityColsPressureTest {

  def main(args: Array[String]): Unit = {
    testRows()
  }

  def generateRows(): Unit = {
    val spark = SparkSession.builder().appName("gen row data").getOrCreate()
//    val spark = Tools.getSparkSession()
    val colsNum = 100
    val k = 1000
    val rowsNumArr = Array(1, 10, 100, 1000, 5000, 10000)
    val fileNames = Array.fill(rowsNumArr.length)("")
    rowsNumArr.indices.foreach { indice =>
      fileNames(indice) = Tools.generateDataSet(rowsNumArr(indice) * k, colsNum, 200, spark)
    }
    Tools.write("./rowsFileName", fileNames)
  }


  def testRows(): Unit = {
    val spark = SparkSession.builder().appName("test row").getOrCreate()
    val rowsNumArr = Array(1, 10, 100, 1000, 5000, 10000)

    val source = Source.fromFile("../DataSet/rowsFileName", "UTF-8")
    val lineIterator =source.getLines
    val out = new PrintWriter("./rowResult")
    var flag = true
    for(file <- lineIterator if flag) {
      val path = "../DataSet/" + file
      out.print(file)
      val df = Tools.readDataFrameASInteger(path, spark).repartition(200)
      df.cache()
      for (i <- 0 to 2) {
        val time = run(df)
        out.print("," + time)
      }
      out.println(",rows:" + df.count() + ",cols:" + df.first().length)
      df.unpersist()
//      flag = false
    }
    out.close
  }


  // 运行FilterLowQualityCols，返回运行时间（秒）
  def run(dataset: Dataset[_]): Double = {
    val names = dataset.schema.names
    val iterationStartTime = System.nanoTime()
    val resultDF = new FilterLowQualityCols()
      .setMissingRatio(0.8)
      .setIDness(0.8)
      .setStability(0.8)
      .setInputCols(names)
      .fit(dataset).transform(dataset)
    (System.nanoTime() - iterationStartTime) / 1e9
  }
}
