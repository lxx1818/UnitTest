package AutoFE.KDD14

import com.common.Tools

object KDD14Test {

  def main(args: Array[String]): Unit = {

    val spark = Tools.getSparkSession()

    val path = ""

    //
    val projectsDF = Tools.read_csv(spark, path + "projects.csv")
    val outcomeDF = Tools.read_csv(spark, path+"outcomes.csv")
    val resourcesDF = Tools.read_csv(spark, path+"resources.csv")
    val essaysDF = Tools.read_csv(spark, path+"essays.csv")

    // first 将outcome中label join 到projects中

    val newProjectsDF = projectsDF.join(outcomeDF)

    // 分离测试集和训练集

    


  }

}
