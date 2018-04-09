/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.network.shuffle.protocol;

import com.google.common.base.Objects;
import io.netty.buffer.ByteBuf;
import org.apache.spark.network.protocol.Encoders;

/**
 * Initial registration message between an executor and its local shuffle server.
 * Returns nothing (empty byte array).
 */
public class UnRegisterExecutor extends BlockTransferMessage {
  public final String appId;
  public final String execId;

  public UnRegisterExecutor(
      String appId,
      String execId) {
    this.appId = appId;
    this.execId = execId;
  }

  @Override
  protected Type type() { return Type.UNREGISTER_EXECUTOR; }

  @Override
  public int hashCode() {
    return Objects.hashCode(appId, execId);
  }

  @Override
  public String toString() {
    return Objects.toStringHelper(this)
      .add("appId", appId)
      .add("execId", execId)
      .toString();
  }

  @Override
  public boolean equals(Object other) {
    if (other != null && other instanceof UnRegisterExecutor) {
      UnRegisterExecutor o = (UnRegisterExecutor) other;
      return Objects.equal(appId, o.appId)
        && Objects.equal(execId, o.execId);
    }
    return false;
  }

  @Override
  public int encodedLength() {
    return Encoders.Strings.encodedLength(appId)
      + Encoders.Strings.encodedLength(execId);
  }

  @Override
  public void encode(ByteBuf buf) {
    Encoders.Strings.encode(buf, appId);
    Encoders.Strings.encode(buf, execId);
  }

  public static UnRegisterExecutor decode(ByteBuf buf) {
    String appId = Encoders.Strings.decode(buf);
    String execId = Encoders.Strings.decode(buf);
    return new UnRegisterExecutor(appId, execId);
  }
}
