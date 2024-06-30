/* ====================================================================
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 * ====================================================================
 */

package org.apache.pylucene.util;

import java.util.Set;
import java.util.Collection;
import java.util.Iterator;
import java.lang.reflect.Array;


public class PythonSet implements Set<Object> {

    private long pythonObject;

    public PythonSet()
    {
    }

    public void pythonExtension(long pythonObject)
    {
        this.pythonObject = pythonObject;
    }
    public long pythonExtension()
    {
        return this.pythonObject;
    }

    public void finalize()
        throws Throwable
    {
        pythonDecRef();
    }

    public native void pythonDecRef();

    @Override
    public native boolean add(Object obj);
    @Override
    public native boolean addAll(Collection<? extends Object> c);
    @Override
    public native void clear();
    @Override
    public native boolean contains(Object obj);
    @Override
    public native boolean containsAll(Collection<? extends Object> c);
    @Override
    public native boolean isEmpty();
    @Override
    public native Iterator<Object> iterator();
    @Override
    public native boolean remove(Object obj);
    @Override
    public native boolean removeAll(Collection<? extends Object> c);
    @Override
    public native boolean retainAll(Collection<? extends Object> c);
    @Override
    public native int size();

    @Override
    public native Object[] toArray();
    
    @SuppressWarnings("unchecked")
    @Override
    public <T> T[] toArray(T[] a)
    {
        Object[] array = toArray();

        if (a.length < array.length) {
            a = (T[]) Array.newInstance(a.getClass().getComponentType(), array.length);
        }

        System.arraycopy(array, 0, a, 0, array.length);

        return a;
    }
}
