package com.znv.spark.mllib

import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.linalg
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

/**
  * 模型的使用
  */
object Demo03ModelUse {
  def main (args: Array[String]): Unit = {
    //1、构建spark环境
    val spark: SparkSession = SparkSession
      .builder()
      .appName("person")
      .master("local")
      .config("spark,sql.shuffle.partitions", "2")
      .getOrCreate()

    import spark.implicits._
    import org.apache.spark.sql.functions._

    //2、加载模型
    val model: LogisticRegressionModel = LogisticRegressionModel.load("data/personModel")

    /**
      * 一条数据
      * 0 1:5.3 2:3.5 3:2.5 4:106.4 5:67.5 6:69.1 7:83
      */
    val vector: linalg.Vector = Vectors.dense(Array(5.3,3.5,2.5,106.4,67.5,69.1,83))

    //3、预测
    val result: Double = model.predict(vector)

    println(result) //打印预测结果

  }
}
