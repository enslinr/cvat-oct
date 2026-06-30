// Copyright (C) CVAT.ai Corporation
//
// SPDX-License-Identifier: MIT

import { BaseShapesAction, ShapesActionInput, ShapesActionOutput } from './base-shapes-action';
import { ActionParameters, ActionParameterType } from './base-action';
import { ShapeType } from '../enums';
import { SerializedShape } from '../server-response-types';
import { Job, Task } from '../session';
import ObjectState from '../object-state';

/**
 * Synchronizes shared boundaries between adjacent polygons.
 * Finds nearby points across all polygons and snaps them to identical coordinates.
 * This is useful after manual editing to restore shared boundary alignment.
 */
export class SyncSharedBoundaries extends BaseShapesAction {
    private snapTolerance: number = 2.0;

    public async init(instance: Job | Task, parameters: Record<string, string | number>): Promise<void> {
        if (parameters?.snapTolerance) {
            this.snapTolerance = Number(parameters.snapTolerance);
        }
    }

    public async destroy(): Promise<void> {
        // nothing to destroy
    }

    public async run(input: ShapesActionInput): Promise<ShapesActionOutput> {
        const { collection, onProgress } = input;
        const created: SerializedShape[] = [];
        const deleted: SerializedShape[] = [];

        const polygons = collection.shapes.filter((shape) => shape.type === ShapeType.POLYGON);
        const total = polygons.length;

        console.log('='.repeat(60));
        console.log('[SyncSharedBoundaries] Synchronizing shared boundaries');
        console.log(`[SyncSharedBoundaries] Found ${total} polygons to process`);
        console.log('='.repeat(60));

        if (total < 2) {
            onProgress('Need at least 2 polygons to sync boundaries', 100);
            return { created: { shapes: [] }, deleted: { shapes: [] } };
        }

        onProgress('Detecting shared points...', 20);

        // Convert polygons to point arrays
        const polygonPoints: Array<Array<{x: number, y: number}>> = polygons.map(
            (poly) => this.flatToPoints(poly.points ?? []),
        );

        // Detect shared points across all polygons
        const sharedPointsMap = this.detectSharedPoints(polygonPoints, this.snapTolerance);

        if (sharedPointsMap.size === 0) {
            onProgress('No shared points found within tolerance', 100);
            return { created: { shapes: [] }, deleted: { shapes: [] } };
        }

        console.log(`[SyncSharedBoundaries] Found ${sharedPointsMap.size} shared point groups`);

        onProgress('Synchronizing shared points...', 60);

        // Synchronize shared points to exact coordinates
        this.synchronizeSharedPoints(polygonPoints, sharedPointsMap);

        onProgress('Creating updated polygons...', 80);

        // Create new polygon shapes with synchronized points
        for (let i = 0; i < polygons.length; i++) {
            const originalPoly = polygons[i];
            const syncedPoints = this.pointsToFlat(polygonPoints[i]);

            // Check if points actually changed
            const pointsChanged = !this.arraysEqual(originalPoly.points ?? [], syncedPoints);

            if (pointsChanged) {
                const newPolygon: SerializedShape = {
                    ...originalPoly,
                    points: syncedPoints,
                    id: undefined,
                    clientID: undefined,
                };

                created.push(newPolygon);
                deleted.push(originalPoly);

                console.log(`[SyncSharedBoundaries] Polygon ${i}: points synchronized`);
            }
        }

        console.log(`[SyncSharedBoundaries] Summary:`);
        console.log(`  - Polygons processed: ${total}`);
        console.log(`  - Polygons updated: ${created.length}`);
        console.log(`  - Shared point groups: ${sharedPointsMap.size}`);

        onProgress(`Synchronized ${created.length} polygon(s) with ${sharedPointsMap.size} shared point groups`, 100);

        return {
            created: { shapes: created },
            deleted: { shapes: deleted },
        };
    }

    /**
     * Detect points that are shared between multiple polygons.
     */
    private detectSharedPoints(
        polygons: Array<Array<{x: number, y: number}>>,
        tolerance: number,
    ): Map<string, Array<{polygonIdx: number, pointIdx: number, point: {x: number, y: number}}>> {
        const pointMap = new Map<string, Array<{polygonIdx: number, pointIdx: number, point: {x: number, y: number}}>>();

        for (let polyIdx = 0; polyIdx < polygons.length; polyIdx++) {
            const polygon = polygons[polyIdx];

            for (let pointIdx = 0; pointIdx < polygon.length; pointIdx++) {
                const point = polygon[pointIdx];
                const pointKey = this.pointToKey(point, tolerance);

                if (!pointMap.has(pointKey)) {
                    pointMap.set(pointKey, []);
                }

                pointMap.get(pointKey)!.push({ polygonIdx: polyIdx, pointIdx, point });
            }
        }

        // Filter to only shared points (from different polygons)
        const sharedPoints = new Map<string, Array<{polygonIdx: number, pointIdx: number, point: {x: number, y: number}}>>();
        for (const [key, usages] of pointMap.entries()) {
            const uniquePolygons = new Set(usages.map((u) => u.polygonIdx));
            if (uniquePolygons.size > 1) {
                sharedPoints.set(key, usages);
            }
        }

        return sharedPoints;
    }

    /**
     * Create a string key for a point based on grid-snapped coordinates.
     */
    private pointToKey(point: {x: number, y: number}, tolerance: number): string {
        const gridSize = Math.max(tolerance, 0.5);
        const x = Math.round(point.x / gridSize) * gridSize;
        const y = Math.round(point.y / gridSize) * gridSize;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
    }

    /**
     * Synchronize shared points to have identical coordinates.
     */
    private synchronizeSharedPoints(
        polygons: Array<Array<{x: number, y: number}>>,
        sharedPointsMap: Map<string, Array<{polygonIdx: number, pointIdx: number, point: {x: number, y: number}}>>,
    ): void {
        for (const [, usages] of sharedPointsMap.entries()) {
            if (usages.length < 2) continue;

            // Calculate average position for the shared point
            let sumX = 0;
            let sumY = 0;
            for (const usage of usages) {
                sumX += usage.point.x;
                sumY += usage.point.y;
            }
            const avgX = sumX / usages.length;
            const avgY = sumY / usages.length;

            // Snap to 0.5 grid for consistency
            const canonicalX = Math.round(avgX * 2) / 2;
            const canonicalY = Math.round(avgY * 2) / 2;

            // Update all points to the canonical position
            for (const { polygonIdx, pointIdx } of usages) {
                polygons[polygonIdx][pointIdx] = { x: canonicalX, y: canonicalY };
            }
        }
    }

    /**
     * Convert flat coordinate array to point objects.
     */
    private flatToPoints(flatPoints: number[]): Array<{x: number, y: number}> {
        const points: Array<{x: number, y: number}> = [];
        for (let i = 0; i < flatPoints.length; i += 2) {
            points.push({ x: flatPoints[i], y: flatPoints[i + 1] });
        }
        return points;
    }

    /**
     * Convert point objects to flat coordinate array.
     */
    private pointsToFlat(points: Array<{x: number, y: number}>): number[] {
        const flat: number[] = [];
        for (const pt of points) {
            flat.push(pt.x, pt.y);
        }
        return flat;
    }

    /**
     * Check if two arrays are equal.
     */
    private arraysEqual(a: number[], b: number[]): boolean {
        if (a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > 0.001) return false;
        }
        return true;
    }

    public applyFilter(input: ShapesActionInput): ShapesActionInput['collection'] {
        const { collection } = input;
        // Only process polygon shapes
        return {
            shapes: collection.shapes.filter((shape) => shape.type === ShapeType.POLYGON),
        };
    }

    public isApplicableForObject(objectState: ObjectState): boolean {
        return objectState.shapeType === ShapeType.POLYGON;
    }

    public get name(): string {
        return 'Synchronize shared boundaries';
    }

    public get parameters(): ActionParameters | null {
        return {
            snapTolerance: {
                type: ActionParameterType.NUMBER,
                defaultValue: '2.0',
                values: ['0.5', '10.0', '0.5'],
            },
        };
    }
}
