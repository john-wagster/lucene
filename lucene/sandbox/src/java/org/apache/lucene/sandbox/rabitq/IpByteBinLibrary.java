package org.apache.lucene.sandbox.rabitq;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Paths;

class IpByteBinLibrary {

    static {
        // FIXME: load dynamically by copying from the packaged jar or adding to the `-Djava.library.path`
        // System.loadLibrary("ipbytebin");
        System.load(Paths.get("lucene/sandbox/src/resources/libipbytebin.dylib").toFile().getAbsolutePath());
    }

    static final Linker linker = Linker.nativeLinker();

    //uint32_t ip_byte_bin(uint64_t *q, uint64_t *d, uint32_t B);
    static final String name = "ip_byte_bin";
    static final FunctionDescriptor fdesc = FunctionDescriptor.of(
            ValueLayout.JAVA_INT,
            ValueLayout.ADDRESS.withTargetLayout(MemoryLayout.sequenceLayout(ValueLayout.JAVA_LONG)),
            ValueLayout.ADDRESS.withTargetLayout(MemoryLayout.sequenceLayout(ValueLayout.JAVA_LONG)),
            ValueLayout.JAVA_INT
    );

    static final SymbolLookup libIpByteBin = SymbolLookup.loaderLookup();

    static final MethodHandle ipByteBin = linker.downcallHandle(
            libIpByteBin.find(name).orElseThrow(),
            fdesc);

    public static long ipByteBinNative(MemorySegment qP, MemorySegment dP, int B) {
        try {
            return (int) ipByteBin.invokeExact(qP, dP, B);
        } catch (Throwable e) {
            if (e instanceof Error err) {
                throw err;
            } else if (e instanceof RuntimeException re) {
                throw re;
            } else {
                throw new RuntimeException(e);
            }
        }
    }

    public static long ipByteBinNative(long[] q, long[] d, int B) {
        // FIXME: better handle errors
        try (Arena offHeap = Arena.ofConfined()) {
            MemorySegment qP = offHeap.allocateArray(ValueLayout.JAVA_LONG, q);
            MemorySegment dP = offHeap.allocateArray(ValueLayout.JAVA_LONG, d);

            try {
                return (int) ipByteBin.invokeExact(qP, dP, B);
            } catch (Throwable e) {
                if (e instanceof Error err) {
                    throw err;
                } else if (e instanceof RuntimeException re) {
                    throw re;
                } else {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    public static void main(String[] args) throws Throwable {

        final long[] q = {Long.parseUnsignedLong("16688761124667853854"),
                Long.parseUnsignedLong("497163260834811885"),
                Long.parseUnsignedLong("12507005279674808930"),
                Long.parseUnsignedLong("13087111665333759114"),
                Long.parseUnsignedLong("15638452587932598080"),
                Long.parseUnsignedLong("2991739204561194362"),
                Long.parseUnsignedLong("3512771521892123148"),
                Long.parseUnsignedLong("5395472162670470289"),
                Long.parseUnsignedLong("2939341458507147136"),
                Long.parseUnsignedLong("3354457473913780782"),
                Long.parseUnsignedLong("10853379235556905612"),
                Long.parseUnsignedLong("69651594990785412"),
                Long.parseUnsignedLong("15046977156337221813"),
                Long.parseUnsignedLong("1400330380264985314"),
                Long.parseUnsignedLong("17137753401750863808"),
                Long.parseUnsignedLong("17710630637095188621"),
                Long.parseUnsignedLong("13705368600651310691"),
                Long.parseUnsignedLong("1568402847701567593"),
                Long.parseUnsignedLong("2838998325747400491"),
                Long.parseUnsignedLong("17370683524038553407"),
                Long.parseUnsignedLong("1597291416854587487"),
                Long.parseUnsignedLong("4789493984731514229"),
                Long.parseUnsignedLong("7622394183577410044"),
                Long.parseUnsignedLong("18319483897828177620")
        };

        final long[] d = {Long.decode("0x7fff95378100"),
                Long.decode("0x7fff95378108"),
                Long.decode("0x7fff95378110"),
                Long.decode("0x7fff95378118"),
                Long.decode("0x7fff95378120"),
                Long.decode("0x7fff95378128")
        };

        System.out.println("dist: " + ipByteBinNative(q, d, 384)); // 1186
    }
}