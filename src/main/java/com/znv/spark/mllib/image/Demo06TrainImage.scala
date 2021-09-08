package com.znv.spark.mllib.image

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object Demo06TrainImage {
  def main (args: Array[String]): Unit = {
    //1、构建spark环境
    val spark: SparkSession = SparkSession
      .builder()
      .appName("image")
      .master("local[*]")
      .config("spark,sql.shuffle.partitions", "2")
      .getOrCreate()

    import spark.implicits._
    import org.apache.spark.sql.functions._


    //2、读取数据
    val data: DataFrame = spark
      .read
      .format("libsvm")
      .load("data/images")


    //3、拆分训练集、测试集
    val array: Array[Dataset[Row]] = data.randomSplit(Array(0.7, 0.3))
    val train: Dataset[Row] = array(0) //训练集
    val test: Dataset[Row] = array(1) //测试集


    //4、构建算法
    val regression: LogisticRegression =
      new LogisticRegression() //逻辑回归
        .setFitIntercept(true) //设置截距
        .setMaxIter(100) //设置最大迭代次数


    //5、将数据带入算法训练模型
    val model: LogisticRegressionModel = regression.fit(train)
    val result: DataFrame = model.transform(test)


    //6、计算准确率
    result
      .select(sum(when($"label" === $"prediction", 1).otherwise(0)) / count($"label") as "rate")
      .show()

    //7、保存模型
    model
      .write
      .overwrite()
      .save("data/imageModel")


  }
}
