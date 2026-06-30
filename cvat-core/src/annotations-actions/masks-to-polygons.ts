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
 * Converts mask annotations to polygon annotations with editable points
 * This allows masks created by SAM2 or other methods to be converted into
 * polygons that can be manually adjusted by dragging points
 */
export class MasksToPolygons extends BaseShapesAction {
    private simplificationTolerance: number = 1.5;

    public async init(instance: Job | Task, parameters: Record<string, string | number>): Promise<void> {
        if (parameters?.simplificationTolerance) {
            this.simplificationTolerance = Number(parameters.simplificationTolerance);
        }
    }

    public async destroy(): Promise<void> {
        // nothing to destroy
    }

    public async run(input: ShapesActionInput): Promise<ShapesActionOutput> {
        const { collection, frameData, onProgress } = input;
        const created: SerializedShape[] = [];
        const deleted: SerializedShape[] = [];

        const masks = collection.shapes.filter((shape) => shape.type === ShapeType.MASK);
        const total = masks.length;

        // ==========================================
        // IMPORTANT: This log confirms the UPDATED masks-to-polygons is running
        // ==========================================
        console.log('='.repeat(60));
        console.log('[MasksToPolygons] VERSION 2.0 - SHARED BOUNDARY PRESERVATION');
        console.log(`[MasksToPolygons] Found ${total} masks to process`);
        console.log('='.repeat(60));

        if (total === 0) {
            onProgress('No masks found to convert', 100);
            return { created: { shapes: [] }, deleted: { shapes: [] } };
        }

        // ==========================================
        // BATCH PROCESSING WITH SHARED BOUNDARY PRESERVATION
        // ==========================================

        // Step 1: Extract polygon points from all masks
        onProgress('Extracting contours from masks...', 10);
        console.log(`[MasksToPolygons] Processing ${total} masks with shared boundary preservation`);

        const allPolygonData: Array<{
            mask: SerializedShape,
            points: Array<{x: number, y: number}>,
            valid: boolean
        }> = [];

        for (let i = 0; i < total; i++) {
            const mask = masks[i];

            try {
                // Decode RLE mask to get contour points
                const polygonPoints = this.rleToPolygon(mask.points, frameData.width, frameData.height);

                if (polygonPoints && polygonPoints.length >= 6) {
                    // Convert flat array to point objects
                    const coords = this.flatToPoints(polygonPoints);
                    allPolygonData.push({ mask, points: coords, valid: true });
                    console.log(`[MasksToPolygons] Mask ${i}: extracted ${coords.length} points`);
                } else {
                    console.warn(`[MasksToPolygons] Mask ${i}: not enough points (${polygonPoints?.length || 0})`);
                    allPolygonData.push({ mask, points: [], valid: false });
                }
            } catch (error) {
                console.warn(`[MasksToPolygons] Failed to extract polygon from mask ${mask.clientID || mask.id}:`, error);
                allPolygonData.push({ mask, points: [], valid: false });
            }
        }

        // Count valid polygons
        const validPolygons = allPolygonData.filter(d => d.valid);
        console.log(`[MasksToPolygons] ${validPolygons.length} valid polygons extracted from ${total} masks`);

        if (validPolygons.length === 0) {
            onProgress('No valid polygons could be extracted', 100);
            return { created: { shapes: [] }, deleted: { shapes: [] } };
        }

        // Step 2: Simplify all polygons together with shared boundary preservation
        onProgress('Simplifying polygons with shared boundary preservation...', 50);

        const polygonsToSimplify = validPolygons.map(d => d.points);
        const simplifiedPolygons = this.simplifyWithSharedBoundaries(
            polygonsToSimplify,
            this.simplificationTolerance,
        );

        console.log(`[MasksToPolygons] Simplified ${simplifiedPolygons.length} polygons`);

        // Step 3: Create polygon shapes from simplified results
        onProgress('Creating polygon shapes...', 80);

        let simplifiedIdx = 0;
        for (let i = 0; i < allPolygonData.length; i++) {
            const { mask, valid } = allPolygonData[i];

            if (!valid) {
                // Skip invalid masks
                continue;
            }

            const simplifiedPoints = simplifiedPolygons[simplifiedIdx];
            simplifiedIdx++;

            if (!simplifiedPoints || simplifiedPoints.length < 3) {
                console.warn(`[MasksToPolygons] Simplified polygon ${i} has too few points`);
                continue;
            }

            // Convert point objects back to flat array
            const flatPoints = this.pointsToFlat(simplifiedPoints);

            if (flatPoints.length >= 6) {
                const polygonShape: SerializedShape = {
                    ...mask,
                    type: ShapeType.POLYGON,
                    points: flatPoints,
                    id: undefined, // Remove ID so it's treated as new
                    clientID: undefined, // Will get new client ID when saved
                };

                created.push(polygonShape);
                deleted.push(mask);

                console.log(`[MasksToPolygons] Created polygon with ${simplifiedPoints.length} points from mask ${mask.clientID || mask.id}`);
            }
        }

        // Log summary
        const totalOriginalPoints = validPolygons.reduce((sum, d) => sum + d.points.length, 0);
        const totalSimplifiedPoints = simplifiedPolygons.reduce((sum, p) => sum + p.length, 0);
        const reductionPercent = totalOriginalPoints > 0
            ? ((1 - totalSimplifiedPoints / totalOriginalPoints) * 100).toFixed(1)
            : '0';

        console.log(`[MasksToPolygons] Summary:`);
        console.log(`  - Masks processed: ${total}`);
        console.log(`  - Polygons created: ${created.length}`);
        console.log(`  - Total points: ${totalOriginalPoints} -> ${totalSimplifiedPoints} (${reductionPercent}% reduction)`);
        console.log(`  - Shared boundaries preserved: YES`);

        onProgress(`Converted ${created.length} mask(s) to polygon(s) with preserved shared boundaries`, 100);

        return {
            created: { shapes: created },
            deleted: { shapes: deleted },
        };
    }

    /**
     * Convert a mask shape to a polygon shape
     * The mask points are in RLE format and need to be decoded to contours
     */
    private async convertMaskToPolygon(
        mask: SerializedShape,
        frameData: { width: number; height: number; number: number },
    ): Promise<SerializedShape | null> {
        if (!mask.points || mask.points.length === 0) {
            console.warn('Mask has no points:', mask);
            return null;
        }

        console.log('Converting mask with points:', mask.points.slice(0, 10), '... (length:', mask.points.length, ')');

        // Decode RLE mask to get contour points
        const polygonPoints = this.rleToPolygon(mask.points, frameData.width, frameData.height);

        console.log('RLE decoded to polygon points:', polygonPoints?.length || 0, 'coordinates');

        if (!polygonPoints || polygonPoints.length < 6) {
            // Need at least 3 points (6 coordinates) for a valid polygon
            console.warn('Not enough polygon points generated:', polygonPoints?.length || 0);
            return null;
        }

        // Simplify the polygon to reduce point count while maintaining shape
        const simplifiedPoints = this.simplifyPolygon(polygonPoints, this.simplificationTolerance);

        console.log('Simplified to:', simplifiedPoints.length, 'coordinates (', simplifiedPoints.length / 2, 'points)');

        // Create new polygon shape with same attributes as the mask
        const polygonShape: SerializedShape = {
            ...mask,
            type: ShapeType.POLYGON,
            points: simplifiedPoints,
            id: undefined, // Remove ID so it's treated as new
            clientID: undefined, // Will get new client ID when saved
        };

        console.log('Created polygon shape:', polygonShape);

        return polygonShape;
    }

    /**
     * Convert RLE (Run-Length Encoding) mask to polygon points
     * CVAT RLE format: [rle_counts..., left, top, right, bottom]
     * The last 4 elements are the bounding box coordinates
     */
    private rleToPolygon(rleData: number[], width: number, height: number): number[] {
        if (rleData.length < 6) {
            return [];
        }

        // Extract bounding box from last 4 elements
        const [left, top, right, bottom] = rleData.slice(-4);
        const maskWidth = right - left + 1;
        const maskHeight = bottom - top + 1;

        // Extract RLE counts (everything except last 4 elements)
        const rle = rleData.slice(0, -4);

        console.log('Mask bounds:', { left, top, right, bottom, width: maskWidth, height: maskHeight });
        console.log('RLE data length:', rle.length);

        // Decode RLE to binary mask
        const mask = new Uint8Array(maskWidth * maskHeight);
        let pixelIdx = 0;
        let value = 0;

        for (let i = 0; i < rle.length; i++) {
            const count = rle[i];
            for (let j = 0; j < count && pixelIdx < mask.length; j++) {
                mask[pixelIdx++] = value;
            }
            value = 1 - value; // Toggle between 0 and 1
        }

        console.log('Decoded', pixelIdx, 'pixels into', maskWidth, 'x', maskHeight, 'mask');

        // Find contours using marching squares algorithm
        const contours = this.findContours(mask, maskWidth, maskHeight);

        if (contours.length === 0) {
            return [];
        }

        // Use the largest contour (outer boundary)
        let largestContour = contours[0];
        let maxArea = this.calculateContourArea(largestContour);

        for (let i = 1; i < contours.length; i++) {
            const area = this.calculateContourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
                largestContour = contours[i];
            }
        }

        // Convert contour coordinates to absolute coordinates with grid snapping
        // Use 0.5 grid snapping like the backend to ensure shared boundaries align
        const polygonPoints: number[] = [];
        for (let i = 0; i < largestContour.length; i++) {
            // Snap to 0.5 grid for shared boundary alignment
            const x = Math.round((largestContour[i].x + left) * 2) / 2;
            const y = Math.round((largestContour[i].y + top) * 2) / 2;
            polygonPoints.push(x);
            polygonPoints.push(y);
        }

        console.log(`[MasksToPolygons] Contour snapped to 0.5 grid for shared boundary alignment`);

        return polygonPoints;
    }

    /**
     * Find contours in a binary mask using Moore-Neighbor tracing
     * This finds boundary pixels (pixels with at least one background neighbor)
     */
    private findContours(mask: Uint8Array, width: number, height: number): Array<Array<{x: number, y: number}>> {
        const contours: Array<Array<{x: number, y: number}>> = [];
        const visited = new Uint8Array(width * height);

        // Find all boundary pixels first
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = y * width + x;

                // Skip if not a foreground pixel or already visited
                if (!mask[idx] || visited[idx]) continue;

                // Check if this is a boundary pixel (has at least one background neighbor)
                const isBoundary = this.isBoundaryPixel(mask, width, height, x, y);

                if (isBoundary) {
                    const contour = this.traceBoundary(mask, visited, width, height, x, y);
                    if (contour.length > 2) {
                        contours.push(contour);
                    }
                }
            }
        }

        console.log('Found', contours.length, 'contour(s)');
        return contours;
    }

    /**
     * Check if a pixel is on the boundary (has at least one background neighbor)
     */
    private isBoundaryPixel(mask: Uint8Array, width: number, height: number, x: number, y: number): boolean {
        // Check 4-connected neighbors
        const neighbors = [
            { dx: -1, dy: 0 },  // left
            { dx: 1, dy: 0 },   // right
            { dx: 0, dy: -1 },  // up
            { dx: 0, dy: 1 },   // down
        ];

        for (const { dx, dy } of neighbors) {
            const nx = x + dx;
            const ny = y + dy;

            // Out of bounds or background pixel = boundary
            if (nx < 0 || nx >= width || ny < 0 || ny >= height || !mask[ny * width + nx]) {
                return true;
            }
        }

        return false;
    }

    /**
     * Trace boundary starting from a boundary pixel using Moore-Neighbor tracing
     */
    private traceBoundary(
        mask: Uint8Array,
        visited: Uint8Array,
        width: number,
        height: number,
        startX: number,
        startY: number,
    ): Array<{x: number, y: number}> {
        const contour: Array<{x: number, y: number}> = [];

        // 8-connected neighbors in clockwise order
        const dirs = [
            { x: 1, y: 0 },   // 0: right
            { x: 1, y: 1 },   // 1: down-right
            { x: 0, y: 1 },   // 2: down
            { x: -1, y: 1 },  // 3: down-left
            { x: -1, y: 0 },  // 4: left
            { x: -1, y: -1 }, // 5: up-left
            { x: 0, y: -1 },  // 6: up
            { x: 1, y: -1 },  // 7: up-right
        ];

        let x = startX;
        let y = startY;
        let dir = 0; // Start searching from the right
        const maxSteps = width * height * 2; // Prevent infinite loops
        let steps = 0;

        do {
            const idx = y * width + x;
            visited[idx] = 1;
            contour.push({ x, y });

            // Moore-Neighbor: search in clockwise direction for next boundary pixel
            let found = false;
            for (let i = 0; i < 8; i++) {
                const checkDir = (dir + i) % 8;
                const nx = x + dirs[checkDir].x;
                const ny = y + dirs[checkDir].y;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    const nidx = ny * width + nx;

                    // Check if this is a foreground boundary pixel
                    if (mask[nidx] && this.isBoundaryPixel(mask, width, height, nx, ny)) {
                        // Don't revisit if we've already been here (except for closing the loop)
                        if (!visited[nidx] || (nx === startX && ny === startY && contour.length > 2)) {
                            x = nx;
                            y = ny;
                            dir = (checkDir + 5) % 8; // Turn left from current direction for next search
                            found = true;
                            break;
                        }
                    }
                }
            }

            if (!found) break;
            steps++;

            // Stop if we've returned to start
            if (x === startX && y === startY && contour.length > 2) {
                break;
            }
        } while (steps < maxSteps);

        return contour;
    }

    /**
     * Calculate the area of a contour
     */
    private calculateContourArea(contour: Array<{x: number, y: number}>): number {
        let area = 0;
        for (let i = 0; i < contour.length; i++) {
            const j = (i + 1) % contour.length;
            area += contour[i].x * contour[j].y;
            area -= contour[j].x * contour[i].y;
        }
        return Math.abs(area) / 2;
    }

    /**
     * Simplify polygon using Ramer-Douglas-Peucker algorithm
     */
    private simplifyPolygon(points: number[], tolerance: number): number[] {
        if (points.length <= 6) return points; // Can't simplify less than 3 points

        const coords: Array<{x: number, y: number}> = [];
        for (let i = 0; i < points.length; i += 2) {
            coords.push({ x: points[i], y: points[i + 1] });
        }

        let simplified = this.douglasPeucker(coords, tolerance);

        // Ensure we have at least 3 points for a valid polygon
        while (simplified.length < 3 && tolerance > 0.1) {
            tolerance = tolerance / 2;
            simplified = this.douglasPeucker(coords, tolerance);
        }

        // If still not enough points, just use the original
        if (simplified.length < 3) {
            console.warn('Simplification resulted in too few points, using original');
            simplified = coords;
        }

        console.log('Simplification: original', coords.length, 'points, simplified to', simplified.length, 'points with tolerance', tolerance);

        const result: number[] = [];
        for (const coord of simplified) {
            result.push(coord.x, coord.y);
        }

        return result;
    }

    /**
     * Douglas-Peucker algorithm for polygon simplification
     */
    private douglasPeucker(points: Array<{x: number, y: number}>, tolerance: number): Array<{x: number, y: number}> {
        if (points.length <= 3) return points; // Keep at least 3 points

        let maxDist = 0;
        let maxDistIdx = 0;
        const first = points[0];
        const last = points[points.length - 1];

        for (let i = 1; i < points.length - 1; i++) {
            const dist = this.perpendicularDistance(points[i], first, last);
            if (dist > maxDist) {
                maxDist = dist;
                maxDistIdx = i;
            }
        }

        if (maxDist > tolerance && maxDistIdx > 0) {
            const left = this.douglasPeucker(points.slice(0, maxDistIdx + 1), tolerance);
            const right = this.douglasPeucker(points.slice(maxDistIdx), tolerance);
            return left.slice(0, -1).concat(right);
        }

        // For closed polygons, if we're simplifying too much, keep first, middle, and last points
        if (points.length >= 3) {
            const middleIdx = Math.floor(points.length / 2);
            return [first, points[middleIdx], last];
        }

        return [first, last];
    }

    /**
     * Calculate perpendicular distance from point to line
     */
    private perpendicularDistance(
        point: {x: number, y: number},
        lineStart: {x: number, y: number},
        lineEnd: {x: number, y: number},
    ): number {
        const dx = lineEnd.x - lineStart.x;
        const dy = lineEnd.y - lineStart.y;

        if (dx === 0 && dy === 0) {
            return Math.sqrt((point.x - lineStart.x) ** 2 + (point.y - lineStart.y) ** 2);
        }

        const numerator = Math.abs(dy * point.x - dx * point.y + lineEnd.x * lineStart.y - lineEnd.y * lineStart.x);
        const denominator = Math.sqrt(dx * dx + dy * dy);

        return numerator / denominator;
    }

    // ==========================================
    // SHARED BOUNDARY DETECTION METHODS
    // ==========================================

    /**
     * Detect edges that are shared between multiple polygons.
     * Two edges are considered shared if their endpoints match within tolerance.
     * 
     * @param polygons - Array of polygons, each polygon is an array of {x, y} points
     * @param tolerance - Distance tolerance for matching edges (default 1.0 pixel)
     * @returns Map of edge keys to arrays of polygon/edge usage info
     */
    private detectSharedEdges(
        polygons: Array<Array<{x: number, y: number}>>,
        tolerance: number = 1.0,
    ): Map<string, Array<{polygonIdx: number, edgeIdx: number, reversed: boolean}>> {
        const edgeMap = new Map<string, Array<{polygonIdx: number, edgeIdx: number, reversed: boolean}>>();

        // Build edge map - for each polygon, extract all edges
        for (let polyIdx = 0; polyIdx < polygons.length; polyIdx++) {
            const polygon = polygons[polyIdx];
            if (polygon.length < 3) continue; // Skip invalid polygons

            for (let edgeIdx = 0; edgeIdx < polygon.length; edgeIdx++) {
                const p1 = polygon[edgeIdx];
                const p2 = polygon[(edgeIdx + 1) % polygon.length];

                // Normalize edge direction (smaller point first for consistent hashing)
                const { normalizedEdge, reversed } = this.normalizeEdge(p1, p2);

                // Create edge key with tolerance-based rounding
                const edgeKey = this.edgeToKey(normalizedEdge.p1, normalizedEdge.p2, tolerance);

                // Add to map
                if (!edgeMap.has(edgeKey)) {
                    edgeMap.set(edgeKey, []);
                }

                edgeMap.get(edgeKey)!.push({ polygonIdx: polyIdx, edgeIdx, reversed });
            }
        }

        // Filter to only shared edges (used by 2+ polygons)
        const sharedEdges = new Map<string, Array<{polygonIdx: number, edgeIdx: number, reversed: boolean}>>();
        for (const [key, usages] of edgeMap.entries()) {
            if (usages.length > 1) {
                sharedEdges.set(key, usages);
            }
        }

        console.log(`[SharedBoundary] Found ${sharedEdges.size} shared edges out of ${edgeMap.size} total edges`);

        return sharedEdges;
    }

    /**
     * Normalize edge direction so that the "smaller" point comes first.
     * This ensures consistent hashing regardless of edge direction.
     * 
     * @param p1 - First endpoint
     * @param p2 - Second endpoint
     * @returns Normalized edge with reversed flag
     */
    private normalizeEdge(
        p1: {x: number, y: number},
        p2: {x: number, y: number},
    ): { normalizedEdge: { p1: {x: number, y: number}, p2: {x: number, y: number} }, reversed: boolean } {
        // Compare points lexicographically (x first, then y)
        if (p1.x < p2.x || (p1.x === p2.x && p1.y < p2.y)) {
            return { normalizedEdge: { p1, p2 }, reversed: false };
        }
        return { normalizedEdge: { p1: p2, p2: p1 }, reversed: true };
    }

    /**
     * Create a string key for an edge based on its endpoints.
     * Coordinates are rounded to a grid based on tolerance for fuzzy matching.
     * 
     * @param p1 - First endpoint (should be the "smaller" point after normalization)
     * @param p2 - Second endpoint
     * @param tolerance - Grid size for rounding coordinates
     * @returns String key for the edge
     */
    private edgeToKey(
        p1: {x: number, y: number},
        p2: {x: number, y: number},
        tolerance: number,
    ): string {
        // Round coordinates to grid for tolerance-based matching
        const gridSize = Math.max(tolerance, 0.5); // Minimum grid size of 0.5
        const x1 = Math.round(p1.x / gridSize) * gridSize;
        const y1 = Math.round(p1.y / gridSize) * gridSize;
        const x2 = Math.round(p2.x / gridSize) * gridSize;
        const y2 = Math.round(p2.y / gridSize) * gridSize;

        return `${x1.toFixed(1)},${y1.toFixed(1)}-${x2.toFixed(1)},${y2.toFixed(1)}`;
    }

    /**
     * Detect points that are shared between multiple polygons.
     * A point is considered shared if another polygon has a point at the same location (within tolerance).
     * 
     * @param polygons - Array of polygons
     * @param tolerance - Distance tolerance for matching points (default 1.0 pixel)
     * @returns Map of point keys to arrays of polygon/point usage info
     */
    private detectSharedPoints(
        polygons: Array<Array<{x: number, y: number}>>,
        tolerance: number = 1.0,
    ): Map<string, Array<{polygonIdx: number, pointIdx: number}>> {
        const pointMap = new Map<string, Array<{polygonIdx: number, pointIdx: number}>>();

        // Build point map - for each polygon, record all points
        for (let polyIdx = 0; polyIdx < polygons.length; polyIdx++) {
            const polygon = polygons[polyIdx];

            for (let pointIdx = 0; pointIdx < polygon.length; pointIdx++) {
                const point = polygon[pointIdx];
                const pointKey = this.pointToKey(point, tolerance);

                if (!pointMap.has(pointKey)) {
                    pointMap.set(pointKey, []);
                }

                pointMap.get(pointKey)!.push({ polygonIdx: polyIdx, pointIdx });
            }
        }

        // Filter to only shared points (used by 2+ polygons from different polygon indices)
        const sharedPoints = new Map<string, Array<{polygonIdx: number, pointIdx: number}>>();
        for (const [key, usages] of pointMap.entries()) {
            // Check if points are from different polygons
            const uniquePolygons = new Set(usages.map(u => u.polygonIdx));
            if (uniquePolygons.size > 1) {
                sharedPoints.set(key, usages);
            }
        }

        console.log(`[SharedBoundary] Found ${sharedPoints.size} shared points across ${polygons.length} polygons`);

        return sharedPoints;
    }

    /**
     * Create a string key for a point based on its coordinates.
     * Coordinates are rounded to a grid based on tolerance for fuzzy matching.
     * 
     * @param point - Point coordinates
     * @param tolerance - Grid size for rounding coordinates
     * @returns String key for the point
     */
    private pointToKey(point: {x: number, y: number}, tolerance: number): string {
        const gridSize = Math.max(tolerance, 0.5);
        const x = Math.round(point.x / gridSize) * gridSize;
        const y = Math.round(point.y / gridSize) * gridSize;

        return `${x.toFixed(1)},${y.toFixed(1)}`;
    }

    /**
     * Get all edges from a polygon as an array of point pairs.
     * 
     * @param polygon - Array of points forming the polygon
     * @returns Array of edges, each edge is {p1, p2}
     */
    private getPolygonEdges(
        polygon: Array<{x: number, y: number}>,
    ): Array<{p1: {x: number, y: number}, p2: {x: number, y: number}}> {
        const edges: Array<{p1: {x: number, y: number}, p2: {x: number, y: number}}> = [];

        for (let i = 0; i < polygon.length; i++) {
            edges.push({
                p1: polygon[i],
                p2: polygon[(i + 1) % polygon.length],
            });
        }

        return edges;
    }

    /**
     * Check if two points are equal within a given tolerance.
     * 
     * @param p1 - First point
     * @param p2 - Second point
     * @param tolerance - Distance tolerance
     * @returns true if points are within tolerance distance
     */
    private pointsEqual(
        p1: {x: number, y: number},
        p2: {x: number, y: number},
        tolerance: number,
    ): boolean {
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy) <= tolerance;
    }

    // ==========================================
    // END SHARED BOUNDARY DETECTION METHODS
    // ==========================================

    // ==========================================
    // SHARED BOUNDARY SIMPLIFICATION METHODS
    // ==========================================

    /**
     * Simplify multiple polygons while preserving shared boundaries.
     * Points that are shared between polygons are kept and remain identical.
     * Non-shared portions are simplified independently using Douglas-Peucker.
     * 
     * @param polygons - Array of polygons to simplify
     * @param tolerance - Douglas-Peucker simplification tolerance
     * @returns Array of simplified polygons with preserved shared boundaries
     */
    private simplifyWithSharedBoundaries(
        polygons: Array<Array<{x: number, y: number}>>,
        tolerance: number,
    ): Array<Array<{x: number, y: number}>> {
        if (polygons.length === 0) return [];
        if (polygons.length === 1) {
            // Single polygon - just simplify normally
            return [this.douglasPeucker(polygons[0], tolerance)];
        }

        console.log(`[SharedBoundary] Simplifying ${polygons.length} polygons with shared boundary preservation`);

        // Step 1: Detect shared points across all polygons
        const sharedPointsMap = this.detectSharedPoints(polygons, 1.0);

        // Step 2: Build a set of point keys that must be preserved (shared points)
        const preservedPointKeys = new Set<string>();
        for (const [key] of sharedPointsMap.entries()) {
            preservedPointKeys.add(key);
        }

        console.log(`[SharedBoundary] ${preservedPointKeys.size} shared points will be preserved`);

        // Step 3: Simplify each polygon while preserving shared points
        const simplifiedPolygons: Array<Array<{x: number, y: number}>> = [];

        for (let polyIdx = 0; polyIdx < polygons.length; polyIdx++) {
            const polygon = polygons[polyIdx];
            const simplified = this.simplifyPolygonPreservingPoints(polygon, tolerance, preservedPointKeys);
            simplifiedPolygons.push(simplified);

            console.log(`[SharedBoundary] Polygon ${polyIdx}: ${polygon.length} -> ${simplified.length} points`);
        }

        // Step 4: Synchronize shared points to ensure exact coordinate match
        this.synchronizeSharedPoints(simplifiedPolygons, sharedPointsMap);

        return simplifiedPolygons;
    }

    /**
     * Simplify a polygon while preserving specific points that must not be removed.
     * Uses a modified Douglas-Peucker that treats preserved points as mandatory.
     * 
     * @param polygon - The polygon to simplify
     * @param tolerance - Douglas-Peucker tolerance
     * @param preservedPointKeys - Set of point keys that must be kept
     * @returns Simplified polygon with preserved points intact
     */
    private simplifyPolygonPreservingPoints(
        polygon: Array<{x: number, y: number}>,
        tolerance: number,
        preservedPointKeys: Set<string>,
    ): Array<{x: number, y: number}> {
        if (polygon.length <= 3) return [...polygon];

        // Mark which points must be preserved
        const mustPreserve: boolean[] = polygon.map(point => {
            const key = this.pointToKey(point, 1.0);
            return preservedPointKeys.has(key);
        });

        // Find segments between preserved points and simplify each segment
        const result: Array<{x: number, y: number}> = [];
        const preservedIndices: number[] = [];

        // Find all preserved point indices
        for (let i = 0; i < polygon.length; i++) {
            if (mustPreserve[i]) {
                preservedIndices.push(i);
            }
        }

        // If no preserved points, simplify the whole polygon normally
        if (preservedIndices.length === 0) {
            return this.douglasPeucker(polygon, tolerance);
        }

        // If all points are preserved, return as-is
        if (preservedIndices.length === polygon.length) {
            return [...polygon];
        }

        // Simplify segments between preserved points
        // Handle the polygon as circular (last point connects to first)
        for (let i = 0; i < preservedIndices.length; i++) {
            const startIdx = preservedIndices[i];
            const endIdx = preservedIndices[(i + 1) % preservedIndices.length];

            // Add the starting preserved point
            result.push({ ...polygon[startIdx] });

            // Extract the segment between these two preserved points
            const segment: Array<{x: number, y: number}> = [];

            if (endIdx > startIdx) {
                // Normal case: segment goes from startIdx to endIdx
                for (let j = startIdx + 1; j < endIdx; j++) {
                    segment.push(polygon[j]);
                }
            } else if (endIdx < startIdx) {
                // Wrap-around case: segment goes from startIdx to end, then from 0 to endIdx
                for (let j = startIdx + 1; j < polygon.length; j++) {
                    segment.push(polygon[j]);
                }
                for (let j = 0; j < endIdx; j++) {
                    segment.push(polygon[j]);
                }
            }
            // If endIdx === startIdx, there's only one preserved point (shouldn't happen with length check)

            // Simplify this segment if it has points
            if (segment.length > 0) {
                // Add endpoints for Douglas-Peucker context
                const segmentWithEndpoints = [
                    polygon[startIdx],
                    ...segment,
                    polygon[endIdx],
                ];

                const simplifiedSegment = this.douglasPeuckerSegment(segmentWithEndpoints, tolerance);

                // Add simplified points (excluding first and last which are the preserved endpoints)
                for (let j = 1; j < simplifiedSegment.length - 1; j++) {
                    result.push(simplifiedSegment[j]);
                }
            }
        }

        // Ensure we have at least 3 points
        if (result.length < 3) {
            // Fallback: return preserved points plus some others
            const fallback: Array<{x: number, y: number}> = [];
            for (let i = 0; i < polygon.length; i++) {
                if (mustPreserve[i]) {
                    fallback.push({ ...polygon[i] });
                }
            }
            // If still not enough, add evenly spaced points
            while (fallback.length < 3 && polygon.length >= 3) {
                const idx = Math.floor(polygon.length * fallback.length / 3);
                if (!mustPreserve[idx]) {
                    fallback.push({ ...polygon[idx] });
                }
            }
            return fallback.length >= 3 ? fallback : [...polygon];
        }

        return result;
    }

    /**
     * Douglas-Peucker simplification for a line segment (not closed polygon).
     * The first and last points are always kept.
     * 
     * @param points - Array of points forming the segment
     * @param tolerance - Maximum perpendicular distance tolerance
     * @returns Simplified segment
     */
    private douglasPeuckerSegment(
        points: Array<{x: number, y: number}>,
        tolerance: number,
    ): Array<{x: number, y: number}> {
        if (points.length <= 2) return [...points];

        // Find point with maximum distance from the line
        let maxDist = 0;
        let maxIdx = 0;
        const first = points[0];
        const last = points[points.length - 1];

        for (let i = 1; i < points.length - 1; i++) {
            const dist = this.perpendicularDistance(points[i], first, last);
            if (dist > maxDist) {
                maxDist = dist;
                maxIdx = i;
            }
        }

        // If max distance exceeds tolerance, recursively simplify
        if (maxDist > tolerance) {
            const left = this.douglasPeuckerSegment(points.slice(0, maxIdx + 1), tolerance);
            const right = this.douglasPeuckerSegment(points.slice(maxIdx), tolerance);
            return left.slice(0, -1).concat(right);
        }

        // Otherwise, just keep endpoints
        return [first, last];
    }

    /**
     * Synchronize shared points across all polygons to ensure exact coordinate match.
     * After simplification, shared points may have slightly different coordinates
     * due to floating-point issues. This method makes them exactly equal.
     * 
     * @param polygons - Array of polygons to synchronize (modified in place)
     * @param sharedPointsMap - Map of shared point keys to their occurrences
     */
    private synchronizeSharedPoints(
        polygons: Array<Array<{x: number, y: number}>>,
        sharedPointsMap: Map<string, Array<{polygonIdx: number, pointIdx: number}>>,
    ): void {
        // For each shared point key, find the actual coordinates in the simplified polygons
        // and make them identical (use the average or first occurrence)
        for (const [pointKey, occurrences] of sharedPointsMap.entries()) {
            // Find the points in simplified polygons that match this key
            const matchingPoints: Array<{polyIdx: number, ptIdx: number, point: {x: number, y: number}}> = [];

            for (const { polygonIdx } of occurrences) {
                if (polygonIdx >= polygons.length) continue;

                const polygon = polygons[polygonIdx];
                for (let ptIdx = 0; ptIdx < polygon.length; ptIdx++) {
                    const pt = polygon[ptIdx];
                    const key = this.pointToKey(pt, 1.0);
                    if (key === pointKey) {
                        matchingPoints.push({ polyIdx: polygonIdx, ptIdx, point: pt });
                    }
                }
            }

            // If we found matching points in multiple polygons, synchronize them
            if (matchingPoints.length > 1) {
                // Use the coordinates from the first occurrence as the canonical value
                const canonicalPoint = matchingPoints[0].point;

                for (let i = 1; i < matchingPoints.length; i++) {
                    const { polyIdx, ptIdx } = matchingPoints[i];
                    polygons[polyIdx][ptIdx] = { ...canonicalPoint };
                }
            }
        }

        console.log(`[SharedBoundary] Synchronized ${sharedPointsMap.size} shared point groups`);
    }

    /**
     * Convert flat coordinate array to array of point objects.
     * @param flatPoints - Array of [x1, y1, x2, y2, ...] coordinates
     * @returns Array of {x, y} point objects
     */
    private flatToPoints(flatPoints: number[]): Array<{x: number, y: number}> {
        const points: Array<{x: number, y: number}> = [];
        for (let i = 0; i < flatPoints.length; i += 2) {
            points.push({ x: flatPoints[i], y: flatPoints[i + 1] });
        }
        return points;
    }

    /**
     * Convert array of point objects to flat coordinate array.
     * @param points - Array of {x, y} point objects
     * @returns Array of [x1, y1, x2, y2, ...] coordinates
     */
    private pointsToFlat(points: Array<{x: number, y: number}>): number[] {
        const flat: number[] = [];
        for (const pt of points) {
            flat.push(pt.x, pt.y);
        }
        return flat;
    }

    // ==========================================
    // END SHARED BOUNDARY SIMPLIFICATION METHODS
    // ==========================================

    public applyFilter(input: ShapesActionInput): ShapesActionInput['collection'] {
        const { collection } = input;
        // Only process mask shapes
        return {
            shapes: collection.shapes.filter((shape) => shape.type === ShapeType.MASK),
        };
    }

    public isApplicableForObject(objectState: ObjectState): boolean {
        // Can be applied to individual mask objects
        return objectState.shapeType === ShapeType.MASK;
    }

    public get name(): string {
        return 'Masks to polygons';
    }

    public get parameters(): ActionParameters | null {
        return {
            simplificationTolerance: {
                type: ActionParameterType.NUMBER,
                defaultValue: '1.5',
                values: ['0.5', '10.0', '0.5'],
            },
        };
    }
}
