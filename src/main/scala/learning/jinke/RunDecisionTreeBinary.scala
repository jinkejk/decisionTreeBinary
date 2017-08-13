package learning.jinke

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.joda.time._
import org.jfree.data.category.DefaultCategoryDataset

object RunDecisionTreeBinary {

  /**
    * main函数
    *
    * @param args
    */
  def main(args: Array[String]): Unit = {
    SetLogger()
    val sc = new SparkContext(new SparkConf().setAppName("App").setMaster("local[4]"))
    println("RunDecisionTreeBinary")
    println("==========数据准备阶段===============")
    val (trainData, validationData, testData, categoriesMap) = prepareDate(sc)
    trainData.persist(); validationData.persist(); testData.persist()

    println("==========训练评估阶段===============")
    println()
    print("是否需要进行参数调校 (Y:是  N:否) ? ")
    if (readLine() == "Y") {
      val model = parametersTunning(trainData, validationData)

      println("==========测试阶段===============")
      val auc = evaluateModel(model, testData)
      println("使用testData测试最佳模型,结果 AUC:" + auc)
      println("==========预测数据===============")
      PredictData(sc, model, categoriesMap)
    } else {
      val model = trainEvaluate(trainData, validationData)
      println("==========测试阶段===============")
      val auc = evaluateModel(model, testData)
      println("使用testData测试最佳模型,结果 AUC:" + auc)
      println("==========预测数据===============")
      PredictData(sc, model, categoriesMap)
    }


    //取消缓存
    trainData.unpersist();validationData.unpersist();testData.unpersist()
  }

  /**
    * 准备数据
    *
    * @param sc
    */
  def prepareDate(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint],
    Map[String, Int]) = {
    //----------------------1.导入并转换数据-------------
    //导入数据
    println("开始导入数据......")
    val rawDataWithHeader = sc.textFile("data/train.tsv")
    //基于分区的map操作,第一个参数是分区的索引,若是第一个分区则删除第一个元素,标题行
    val rawData = rawDataWithHeader.mapPartitionsWithIndex {
      (idx, iter) => if (idx == 0) iter.drop(1) else iter
    }
    //tsv是以水平制表符分割的
    val lines = rawData.map(_.split("\t"))
    println("共计:" + lines.count.toString + "条")

    //----------------------2.创建训练评估所需数据 RDD[LabeledPoint]-------------
    //创建训练评估所需的数据
    //第3列是网站分类特征:business,sport... 返回所有的分类,并依据索引号创建map,从0开始数起(business,0)..
    val categoriesMap = lines.map(fields => fields(3)).distinct.collect.zipWithIndex.toMap
    val labeledPointRDD = lines.map { fields =>
      //删除双引号
      val trFields = fields.map(_.replaceAll("\"", ""))

      //固定维数的数组,14维
      val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
      //获取对应的id, (business,0),返回0
      val categoryIdx = categoriesMap(fields(3))
      //对应位置设置维1, (1,0,0,0)
      categoryFeaturesArray(categoryIdx) = 1

      //第4-25列是数字特征列,slice取特定列组成新RDD
      val numericalFeatures = trFields.slice(4, fields.size - 1)
        .map(d => if (d == "?") 0.0 else d.toDouble)

      //最后一列是label:0,1
      val label = trFields(fields.size - 1).toInt


      //label和feature(分类特征+数字特征), 稠密的向量,也就是直接输入向量就好了
      LabeledPoint(label, Vectors.dense(categoryFeaturesArray ++ numericalFeatures))
    }

    //----------------------3.以随机方式将数据分为3个部分并且返回-------------
    //按照8:1:1的比例分割
    val Array(trainData, validationData, testData) = labeledPointRDD.randomSplit(Array(8, 1, 1))

    println("将数据分trainData:" + trainData.count() + "   validationData:" + validationData.count()
      + "   testData:" + testData.count())

    return (trainData, validationData, testData, categoriesMap) //返回数据

  }

  /**
    * 训练评估
    */
  def trainEvaluate(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): DecisionTreeModel = {
    print("开始训练...")
    //评估方法熵或者gini;
    val (model, time) = trainModel(trainData, "entropy", 10, 10)

    println("训练完成,所需时间:" + time + "毫秒")
    val AUC = evaluateModel(model, validationData)
    println("评估结果AUC=" + AUC)
    return (model)
  }

  /**
    * 训练模型
    *
    * @param trainData
    * @param impurity
    * @param maxDepth
    * @param maxBins
    * @return
    */
  def trainModel(trainData: RDD[LabeledPoint], impurity: String, maxDepth: Int, maxBins: Int): (DecisionTreeModel, Double) = {
    val startTime = new DateTime()
    //maxBins:最大分之;2个类;Map[Int, Int](): 分类特征信息
    val model = DecisionTree.trainClassifier(trainData, 2, Map[Int, Int](), impurity, maxDepth, maxBins)
    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)
    (model, duration.getMillis())
  }

  /**
    * 模型评估
    * 二元分类采用ACU评估
    *
    * @param model
    * @param validationData
    * @return
    */
  def evaluateModel(model: DecisionTreeModel, validationData: RDD[LabeledPoint]): (Double) = {
    //data是LabelPoint类型, 保存为(预测结果,真实结果)
    val scoreAndLabels = validationData.map { data =>
      var predict = model.predict(data.features)
      (predict, data.label)
    }

    //计算AUC
    val Metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val AUC = Metrics.areaUnderROC
    //返回AUC
    (AUC)
  }

  /**
    * 预测阶段
    *
    * @param sc
    * @param model         决策树模型
    * @param categoriesMap 所有类别及对应的索引
    */
  def PredictData(sc: SparkContext, model: DecisionTreeModel, categoriesMap: Map[String, Int]): Unit = {
    //----------------------1.导入并转换数据-------------
    val rawDataWithHeader = sc.textFile("data/test.tsv")
    val rawData = rawDataWithHeader.mapPartitionsWithIndex { (idx, iter) => if (idx == 0) iter.drop(1) else iter }
    val lines = rawData.map(_.split("\t"))
    println("共计：" + lines.count.toString() + "条")

    //----------------------2.创建训练评估所需数据 RDD[LabeledPoint]-------------
    //lazy记得collect
    val dataRDD = lines.take(10).map { fields =>
      //去掉引号
      val trFields = fields.map(_.replaceAll("\"", ""))
      val categoryFeaturesArray = Array.ofDim[Double](categoriesMap.size)
      val categoryIdx = categoriesMap(fields(3))
      //创建类别特征(1,0,0,0)
      categoryFeaturesArray(categoryIdx) = 1
      //数字特征
      val numericalFeatures = trFields.slice(4, fields.size)
        .map(d => if (d == "?") 0.0 else d.toDouble)
      val label = 0
      //----------------------3进行预测-------------
      val url = trFields(0)

      //构建预测特征
      val Features = Vectors.dense(categoryFeaturesArray ++ numericalFeatures)
      val predict = model.predict(Features).toInt

      //输出
      var predictDesc = {
        predict match {
          case 0 => "暂时性网页(ephemeral)";
          case 1 => "长青网页(evergreen)";
        }
      }
      println("网址：  " + url + "==>预测:" + predictDesc)
    }

  }

  /**
    * 调整参数
    * 同一时间只调整一个参数
    * @param trainData
    * @param validationData
    * @return
    */
  def parametersTunning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): DecisionTreeModel = {
    println("-----评估 Impurity参数使用 gini, entropy---------")
//    evaluateParameter(trainData, validationData, "impurity", Array("gini", "entropy"), Array(10), Array(10))

    println("-----评估MaxDepth参数使用 (3, 5, 10, 15, 20)---------")
    evaluateParameter(trainData, validationData, "maxDepth", Array("gini"), Array(3, 5, 10, 15, 20, 25), Array(10))

    println("-----评估maxBins参数使用 (3, 5, 10, 50, 100)---------")
    evaluateParameter(trainData, validationData, "maxBins", Array("gini"), Array(10), Array(3, 5, 10, 50, 100, 200))

    println("-----所有参数交叉评估找出最好的参数组合---------")
    val bestModel = evaluateAllParameter(trainData, validationData, Array("gini", "entropy"),
      Array(3, 5, 10, 15, 20), Array(3, 5, 10, 50, 100))
    return (bestModel)
  }

  def evaluateParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint],
                        evaluateParameter: String, impurityArray: Array[String], maxdepthArray: Array[Int], maxBinsArray: Array[Int]) =
  {
    //画图,柱行图
    var dataBarChart = new DefaultCategoryDataset()
    //折线图
    var dataLineChart = new DefaultCategoryDataset()

    //三重循环
    for (impurity <- impurityArray; maxDepth <- maxdepthArray; maxBins <- maxBinsArray) {
      val (model, time) = trainModel(trainData, impurity, maxDepth, maxBins)
      val auc = evaluateModel(model, validationData)

      val parameterData =
        evaluateParameter match {
          case "impurity" => impurity;
          case "maxDepth" => maxDepth;
          case "maxBins"  => maxBins
        }
      dataBarChart.addValue(auc, evaluateParameter, parameterData.toString())
      dataLineChart.addValue(time, "Time", parameterData.toString())
    }
    Chart.plotBarLineChart("DecisionTree evaluations " + evaluateParameter, evaluateParameter, "AUC", 0.58, 0.7, "Time", dataBarChart, dataLineChart)
  }

  /**
    * 对3个参数进行交叉评估
    * @param trainData
    * @param validationData
    * @param impurityArray
    * @param maxdepthArray
    * @param maxBinsArray
    * @return
    */
  def evaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], impurityArray: Array[String], maxdepthArray: Array[Int], maxBinsArray: Array[Int]): DecisionTreeModel =
  {
    val evaluationsArray =
      for (impurity <- impurityArray; maxDepth <- maxdepthArray; maxBins <- maxBinsArray) yield {
        val (model, time) = trainModel(trainData, impurity, maxDepth, maxBins)
        val auc = evaluateModel(model, validationData)
        (impurity, maxDepth, maxBins, auc)
      }

    val BestEval = (evaluationsArray.sortBy(_._4).reverse)(0)
    println("调校后最佳参数：impurity:" + BestEval._1 + "  ,maxDepth:" + BestEval._2 + "  ,maxBins:" + BestEval._3
      + "  ,结果AUC = " + BestEval._4)
    val (bestModel, time) = trainModel(trainData.union(validationData), BestEval._1, BestEval._2, BestEval._3)
    return bestModel
  }

  /**
    * 设置输出格式
    *
    * @return
    */
  def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }
}
