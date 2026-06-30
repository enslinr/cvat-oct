// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

/**
 * SharedBoundaryHandler - Detects and manages shared boundary points across polygons
 * 
 * This handler enables linked editing of polygon points that lie on shared boundaries.
 * When a point is moved on one polygon, all linked points on adjacent polygons
 * are automatically updated to maintain boundary alignment.
 */

export interface SharedPointInfo {
    polygonClientID: number;
    pointIndex: number;
    x: number;
    y: number;
}

export interface LinkedPointGroup {
    key: string;  // Point key for identification
    points: SharedPointInfo[];
}

export interface SharedBoundaryHandler {
    /**
     * Analyze all polygons on the current frame and detect shared points
     * @param states - Array of annotation states (polygons)
     * @param tolerance - Distance tolerance for considering points as shared (default 1.0 pixel)
     */
    analyzeSharedPoints(states: any[], tolerance?: number): void;

    /**
     * Get all points linked to a specific polygon point
     * @param clientID - The polygon's client ID
     * @param pointIndex - The index of the point in the polygon
     * @returns Array of linked point information, or empty array if not shared
     */
    getLinkedPoints(clientID: number, pointIndex: number): SharedPointInfo[];

    /**
     * Check if a point is shared with other polygons
     * @param clientID - The polygon's client ID
     * @param pointIndex - The index of the point in the polygon
     * @returns true if the point is shared
     */
    isSharedPoint(clientID: number, pointIndex: number): boolean;

    /**
     * Get all shared point groups (for visualization)
     * @returns Array of linked point groups
     */
    getAllSharedGroups(): LinkedPointGroup[];

    /**
     * Update linked points when a point is moved
     * @param clientID - The polygon's client ID
     * @param pointIndex - The index of the point being moved
     * @param newX - New X coordinate
     * @param newY - New Y coordinate
     * @returns Array of updates to apply to other polygons
     */
    propagatePointMove(
        clientID: number,
        pointIndex: number,
        newX: number,
        newY: number,
    ): Array<{ clientID: number; pointIndex: number; x: number; y: number }>;

    /**
     * Clear all shared point data
     */
    clear(): void;
}

export class SharedBoundaryHandlerImpl implements SharedBoundaryHandler {
    // Map: clientID -> Map<pointIndex, pointKey>
    private pointToKeyMap: Map<number, Map<number, string>>;

    // Map: pointKey -> LinkedPointGroup
    private sharedGroups: Map<string, LinkedPointGroup>;

    // Tolerance for point matching
    private tolerance: number;

    constructor() {
        this.pointToKeyMap = new Map();
        this.sharedGroups = new Map();
        this.tolerance = 1.0;
    }

    /**
     * Create a string key for a point based on its coordinates
     * Coordinates are rounded to a grid based on tolerance for fuzzy matching
     */
    private pointToKey(x: number, y: number): string {
        const gridSize = Math.max(this.tolerance, 0.5);
        const roundedX = Math.round(x / gridSize) * gridSize;
        const roundedY = Math.round(y / gridSize) * gridSize;
        return `${roundedX.toFixed(1)},${roundedY.toFixed(1)}`;
    }

    public analyzeSharedPoints(states: any[], tolerance: number = 1.0): void {
        this.clear();
        this.tolerance = tolerance;

        // Filter to only polygon shapes
        const polygons = states.filter(
            (state) => state.shapeType === 'polygon' && !state.hidden && !state.outside,
        );

        if (polygons.length < 2) {
            console.log('[SharedBoundary] Less than 2 visible polygons, no shared points possible');
            return;
        }

        // Temporary map to collect all points by key
        const pointsByKey = new Map<string, SharedPointInfo[]>();

        // Process each polygon
        for (const polygon of polygons) {
            const { clientID, points } = polygon;
            const pointMap = new Map<number, string>();

            // Points are stored as flat array [x1, y1, x2, y2, ...]
            for (let i = 0; i < points.length; i += 2) {
                const x = points[i];
                const y = points[i + 1];
                const pointIndex = i / 2;
                const key = this.pointToKey(x, y);

                pointMap.set(pointIndex, key);

                if (!pointsByKey.has(key)) {
                    pointsByKey.set(key, []);
                }

                pointsByKey.get(key)!.push({
                    polygonClientID: clientID,
                    pointIndex,
                    x,
                    y,
                });
            }

            this.pointToKeyMap.set(clientID, pointMap);
        }

        // Filter to only shared points (points from different polygons)
        for (const [key, points] of pointsByKey.entries()) {
            // Get unique polygon IDs
            const uniquePolygons = new Set(points.map((p) => p.polygonClientID));

            if (uniquePolygons.size > 1) {
                // This is a shared point
                this.sharedGroups.set(key, {
                    key,
                    points,
                });
            }
        }

        console.log(
            `[SharedBoundary] Analyzed ${polygons.length} polygons, found ${this.sharedGroups.size} shared point groups`,
        );
    }

    public getLinkedPoints(clientID: number, pointIndex: number): SharedPointInfo[] {
        const pointMap = this.pointToKeyMap.get(clientID);
        if (!pointMap) {
            return [];
        }

        const key = pointMap.get(pointIndex);
        if (!key) {
            return [];
        }

        const group = this.sharedGroups.get(key);
        if (!group) {
            return [];
        }

        // Return all points except the one being queried
        return group.points.filter(
            (p) => !(p.polygonClientID === clientID && p.pointIndex === pointIndex),
        );
    }

    public isSharedPoint(clientID: number, pointIndex: number): boolean {
        const pointMap = this.pointToKeyMap.get(clientID);
        if (!pointMap) {
            return false;
        }

        const key = pointMap.get(pointIndex);
        if (!key) {
            return false;
        }

        return this.sharedGroups.has(key);
    }

    public getAllSharedGroups(): LinkedPointGroup[] {
        return Array.from(this.sharedGroups.values());
    }

    public propagatePointMove(
        clientID: number,
        pointIndex: number,
        newX: number,
        newY: number,
    ): Array<{ clientID: number; pointIndex: number; x: number; y: number }> {
        const linkedPoints = this.getLinkedPoints(clientID, pointIndex);

        return linkedPoints.map((point) => ({
            clientID: point.polygonClientID,
            pointIndex: point.pointIndex,
            x: newX,
            y: newY,
        }));
    }

    public clear(): void {
        this.pointToKeyMap.clear();
        this.sharedGroups.clear();
    }
}

// Singleton instance for use across the canvas
let sharedBoundaryHandlerInstance: SharedBoundaryHandler | null = null;

export function getSharedBoundaryHandler(): SharedBoundaryHandler {
    if (!sharedBoundaryHandlerInstance) {
        sharedBoundaryHandlerInstance = new SharedBoundaryHandlerImpl();
    }
    return sharedBoundaryHandlerInstance;
}

export function resetSharedBoundaryHandler(): void {
    if (sharedBoundaryHandlerInstance) {
        sharedBoundaryHandlerInstance.clear();
    }
    sharedBoundaryHandlerInstance = null;
}
