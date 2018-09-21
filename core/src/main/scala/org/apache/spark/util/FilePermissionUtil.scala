package org.apache.spark.util

import java.io.File
import java.nio.file.attribute.PosixFilePermissions
import java.nio.file.{Files, Paths}

import org.apache.spark.internal.Logging

object FilePermissionUtil extends Logging {
  val allPermissions = PosixFilePermissions.fromString("rwxrwxrwx");

  def setAllPermission(file : File): Unit = {
    if (file.exists()) {
      logInfo("修改文件/目录权限：" + file.getAbsolutePath)
      try {
        Files.setPosixFilePermissions(Paths.get(file.getAbsolutePath), allPermissions)
      } catch {
        case e : Exception =>
          logError("修改文件/目录权限失败！", e)
      }
    }
  }
}
