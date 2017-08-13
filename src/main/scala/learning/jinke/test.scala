package learning.jinke

object test {

  def main(args: Array[String]): Unit = {
    val one = Array[Double](1,0,0,0)
    val two = Array[Double](21.2,23,4,54,554.54,465.65,0.22444,242,42.424)

    println((one ++ two).mkString(" "))
    println(fun01(3))
  }

  def fun01(str: Int): String ={
    return str match {
      case 0 => "one";
      case 1 => "two";
      case 2 => "three";
      case 3 => "four";
    }
  }
}
