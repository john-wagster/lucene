package org.apache.lucene.sandbox.rabitq;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;

public class IOUtils {

    public static float[][] readFvecs(FileInputStream stream) throws IOException {
        FileChannel fc = stream.getChannel();

        ByteBuffer bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        fc.read(bb);
        bb.flip();
        int dimensions = bb.getInt();

        long fsize = fc.size();
        int size = (int) ((fsize) / (dimensions * 4 + 4));
        float[][] data = new float[size][dimensions];
        for (int i = 0; i < size; i++) {
            if (i != 0) {
                bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
                fc.read(bb);
                bb.flip();
                assert dimensions == bb.getInt();    /// dimensions every time
            }

            bb = ByteBuffer.allocate(dimensions*4).order(ByteOrder.LITTLE_ENDIAN);
            fc.read(bb);
            bb.flip();
            bb.asFloatBuffer().get(data[i]);
        }

        return data;
    }

    public static int[][] readIvecs(FileInputStream stream) throws IOException {
        FileChannel fc = stream.getChannel();
        ByteBuffer bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
        fc.read(bb);
        bb.flip();
        int dimensions = bb.getInt();

        long fsize = fc.size();
        int size = (int) ((fsize) / (dimensions * 4 + 4));
        int[][] data = new int[size][dimensions];
        for (int i = 0; i < size; i++) {
            if (i != 0) {
                bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
                fc.read(bb);
                bb.flip();
                assert dimensions == bb.getInt();    /// dimensions every time
            }

            bb = ByteBuffer.allocate(dimensions*4).order(ByteOrder.LITTLE_ENDIAN);
            fc.read(bb);
            bb.flip();
            bb.asIntBuffer().get(data[i]);
        }

        return data;
    }

    public static void toFvecs(FileOutputStream stream, float[][] data) throws IOException {
        FileChannel fc = stream.getChannel();
        int dimensions = data[0].length;
        ByteBuffer bb;

        for (int i = 0; i < data.length; i++) {
            bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            bb.putInt(data[0].length);
            bb.flip();
            fc.write(bb);

            bb = ByteBuffer.allocate(dimensions*4).order(ByteOrder.LITTLE_ENDIAN);
            for(float d : data[i]) {
                bb.putFloat(d);
            }
            bb.flip();
            fc.write(bb);
        }
    }

    public static void toIvecs(FileOutputStream stream, int[][] data) throws IOException {
        FileChannel fc = stream.getChannel();
        int dimensions = data[0].length;
        ByteBuffer bb;

        for (int i = 0; i < data.length; i++) {
            bb = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            bb.putInt(data[0].length);
            bb.flip();
            fc.write(bb);

            bb = ByteBuffer.allocate(dimensions*4).order(ByteOrder.LITTLE_ENDIAN);
            for(int d : data[i]) {
                bb.putInt(d);
            }
            bb.flip();
            fc.write(bb);
        }
    }
}

