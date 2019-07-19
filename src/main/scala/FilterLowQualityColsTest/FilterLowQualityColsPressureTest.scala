package FilterLowQualityColsTest

import java.io.{BufferedWriter, FileOutputStream, OutputStreamWriter, PrintWriter}

import com.common.Tools
import org.apache.spark.ml.preprocess.FilterLowQualityCols
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.io.Source


object FilterLowQualityColsPressureTest {

  def main(args: Array[String]): Unit = {
    generateCols()
  }

  def generateRows(): Unit = {
    val spark = SparkSession.builder().appName("gen rows data").getOrCreate()
//    val spark = Tools.getSparkSession()
    val colsNum = 10
    val k = 1000
    val rowsNumArr = Array(1, 10, 100, 1000, 2000, 4000, 6000, 10000)
    val fileNames = Array.fill(rowsNumArr.length)("")
    rowsNumArr.indices.foreach { indice =>
      fileNames(indice) = Tools.generateDataSet(rowsNumArr(indice) * k, colsNum, 100, spark, path = "./rows/")
    }
    Tools.write("./rowsFileName", fileNames)
  }

  def generateCols(): Unit = {
    val spark = SparkSession.builder().appName("gen cols data").getOrCreate()
    //    val spark = Tools.getSparkSession()
    val colsNumArr = Array(10, 20, 40, 80, 200, 400, 600, 1000)
    val k = 1000
    val rowsNum = 10000 * k
    val fileNames = Array.fill(colsNumArr.length)("")
    colsNumArr.indices.foreach { indice =>
      fileNames(indice) = Tools.generateDataSet(rowsNum, colsNumArr(indice), 100, spark, path = "./cols/")
    }
    Tools.write("./colsFileName", fileNames)
  }


  def testRows(): Unit = {
    val spark = SparkSession.builder().appName("test rows").getOrCreate()

    val source = Source.fromFile("../DataSet/rowsFileName", "UTF-8")
    val lineIterator =source.getLines
    var flag = true
    for(file <- lineIterator if flag) {
      val path = "../DataSet/rows/" + file
      val out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream("./rowResult", true))), false)
      out.print(file)
      val df = Tools.readDataFrameASInteger(path, spark).repartition(100)
      df.cache()
      df.first()
      for (i <- 0 to 4) {
        val time = run(df)
        out.print("," + time)
      }
      out.println(",rows:" + df.count() + ",cols:" + df.first().length)
      df.unpersist()
//      flag = false
      out.close
    }
  }

  def testCols(): Unit = {
    val spark = SparkSession.builder().appName("test cols").getOrCreate()

    val source = Source.fromFile("../DataSet/colsFileName", "UTF-8")
    val lineIterator =source.getLines
    var flag = true
    for(file <- lineIterator if flag) {
      val path = "../DataSet/cols/" + file
      val out = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream("./colResult", true))), false)
      out.print(file)
      val df = Tools.readDataFrameASInteger(path, spark).repartition(100)
      df.cache()
      df.first()
      for (i <- 0 to 4) {
        val time = run(df)
        out.print("," + time)
      }
      out.println(",rows:" + df.count() + ",cols:" + df.first().length)
      df.unpersist()
      //      flag = false
      out.close
    }
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
