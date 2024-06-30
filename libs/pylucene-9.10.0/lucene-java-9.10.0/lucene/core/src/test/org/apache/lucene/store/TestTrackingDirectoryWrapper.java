/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.store;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import org.apache.lucene.tests.store.BaseDirectoryTestCase;

public class TestTrackingDirectoryWrapper extends BaseDirectoryTestCase {

  @Override
  protected Directory getDirectory(Path path) throws IOException {
    return new TrackingDirectoryWrapper(new ByteBuffersDirectory());
  }

  public void testTrackEmpty() throws IOException {
    TrackingDirectoryWrapper dir = new TrackingDirectoryWrapper(new ByteBuffersDirectory());
    assertEquals(Collections.emptySet(), dir.getCreatedFiles());
  }

  public void testTrackCreate() throws IOException {
    TrackingDirectoryWrapper dir = new TrackingDirectoryWrapper(new ByteBuffersDirectory());
    dir.createOutput("foo", newIOContext(random())).close();
    assertEquals(asSet("foo"), dir.getCreatedFiles());
  }

  public void testTrackDelete() throws IOException {
    TrackingDirectoryWrapper dir = new TrackingDirectoryWrapper(new ByteBuffersDirectory());
    dir.createOutput("foo", newIOContext(random())).close();
    assertEquals(asSet("foo"), dir.getCreatedFiles());
    dir.deleteFile("foo");
    assertEquals(Collections.emptySet(), dir.getCreatedFiles());
  }

  public void testTrackRename() throws IOException {
    TrackingDirectoryWrapper dir = new TrackingDirectoryWrapper(new ByteBuffersDirectory());
    dir.createOutput("foo", newIOContext(random())).close();
    assertEquals(asSet("foo"), dir.getCreatedFiles());
    dir.rename("foo", "bar");
    assertEquals(asSet("bar"), dir.getCreatedFiles());
  }

  public void testTrackCopyFrom() throws IOException {
    TrackingDirectoryWrapper source = new TrackingDirectoryWrapper(new ByteBuffersDirectory());
    TrackingDirectoryWrapper dest = new TrackingDirectoryWrapper(new ByteBuffersDirectory());
    source.createOutput("foo", newIOContext(random())).close();
    assertEquals(asSet("foo"), source.getCreatedFiles());
    dest.copyFrom(source, "foo", "bar", newIOContext(random()));
    assertEquals(asSet("bar"), dest.getCreatedFiles());
    assertEquals(asSet("foo"), source.getCreatedFiles());
  }
}
