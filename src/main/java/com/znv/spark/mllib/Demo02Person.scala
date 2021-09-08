package com.znv.spark.mllib


import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object Demo02Person {
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

    //2、读取被预处理后的人体SVM格式的数据
    val data: DataFrame = spark
      .read
      .format("libsvm")
      .load("data/人体指标.txt")

    //data.show(false) //查看数据，false可以看全


    //3、将数据拆分为训练集和测试集：训练集0.7，测试集0.3
    val splitDS: Array[Dataset[Row]] = data.randomSplit(Array(0.7, 0.3))
    val tran: Dataset[Row] = splitDS(0) //训练集
    val test: Dataset[Row] = splitDS(1) //测试集


    //4、选择算法，这里是二分类问题，可以采用逻辑回归算法
    val regression: LogisticRegression = new LogisticRegression()
      .setFitIntercept(true) //设置截距
      .setMaxIter(100) //设置最大迭代次数


    //5、将训练集带入算法训练模型
    val model: LogisticRegressionModel = regression.fit(tran)


    //6、模型的评估：会在最后增加一列表示预测的结果
    val testDF: DataFrame = model.transform(test)

    //testDF.show(10000,false) //查看数据，10000表示行数

    //7、对预测的结果和表中原来的结果继续比较，如果相等置1相加，不相等置0，最后求出准备的比例(相等的/总数)
    val p: DataFrame = testDF.select(sum(when($"label" === $"prediction", value = "1.0").otherwise(value = "0.0")) / count($"label") as "p")

    //p.show() //查看准确率


    //8、如果模型准确率达标，将模型保存
    model.save("data/personModel")


    //9、模型的使用:找到路径下保存的模型直接使用
    //val model: LogisticRegressionModel = LogisticRegressionModel.load("data/personModel")


  }
}
