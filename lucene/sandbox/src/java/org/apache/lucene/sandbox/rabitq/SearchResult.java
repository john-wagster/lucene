package org.apache.lucene.sandbox.rabitq;

import java.util.Objects;

public class SearchResult {
    private float distToCentroid;
    private int clusterId;

    public SearchResult(float distToCentroid, int clusterId) {
        this.distToCentroid = distToCentroid;
        this.clusterId = clusterId;
    }

    public float getDistToCentroid() {
        return distToCentroid;
    }

    public int getClusterId() {
        return clusterId;
    }

    public void setDistToCentroid(float distToCentroid) {
        this.distToCentroid = distToCentroid;
    }

    public void setClusterId(int clusterId) {
        this.clusterId = clusterId;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        SearchResult that = (SearchResult) o;
        return Float.compare(distToCentroid, that.distToCentroid) == 0 && clusterId == that.clusterId;
    }

    @Override
    public int hashCode() {
        return Objects.hash(distToCentroid, clusterId);
    }
}
