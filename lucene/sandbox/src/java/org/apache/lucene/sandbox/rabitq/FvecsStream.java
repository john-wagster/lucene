package org.apache.lucene.sandbox.rabitq;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public class FvecsStream {
    private final FileInputStream stream;
    private final FileChannel fc;
    private final int size;
    private final int dimensions;

    private final int cacheSize;
    private final float[][] cache;
    private int offset;
    private int cacheOffset;
    private int actualCacheSize;

    public FvecsStream(FileInputStream stream, int dimensions, int cacheSize) throws IOException {
        this.stream = stream;
        this.fc = this.stream.getChannel();
        long fsize = fc.size();
        size = (int) ((fsize) / (dimensions * 4 + 4));
        this.dimensions = dimensions;

        this.cacheSize = Math.min(cacheSize, this.size);;
        this.cache = new float[cacheSize][dimensions];
        this.offset = 0;
        this.cacheOffset = this.cacheSize;  // invalidate the cache
    }

    public float[] getNextFvec() throws IOException {
        // FIXME: throw exception when size - offset < 0 --- out of elements
        if(cacheOffset < cacheSize) {
            float[] vector = cache[cacheOffset];
            cacheOffset++;
            offset++;
            return vector;
        } else {
            int actualCacheSize = Math.min(size - offset, cacheSize);

            // FIXME: read this as one contiguous block
            for (int i = 0; i < actualCacheSize; i++) {
                ByteBuffer bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
                fc.read(bb);
                bb.flip();

                bb = ByteBuffer.allocate(dimensions * 4).order(ByteOrder.LITTLE_ENDIAN);
                fc.read(bb);
                bb.flip();
                bb.asFloatBuffer().get(cache[i]);
            }

            cacheOffset = 1;
            offset++;
            return cache[0];
        }
    }

    public int getTotalFvecs() {
        return this.size;
    }

    public void close() throws IOException {
        this.fc.close();
        this.stream.close();
    }
}
