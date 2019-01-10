package com.common

import org.apache.spark.ml.GeneralFunSuite

class ToolsTest extends GeneralFunSuite{


  test("kFold") {
    val data = OftenUseDataSet.getIrisDataSet(spark)
    val arr = Tools.kFold(data)

    print("OK")
  }

}
