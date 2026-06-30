# Masks to Polygons Converter - Implementation Guide

## Overview

This feature adds a "Masks to polygons" action under **Menu → Run actions** that converts mask annotations (created by SAM2 or other segmentation tools) into polygon annotations with editable points.

## What It Does

- **Converts masks to polygons**: Transforms raster mask data into vector polygon data
- **Makes annotations editable**: The resulting polygons have draggable points for manual refinement
- **Preserves annotation attributes**: Maintains labels, colors, and other properties from the original mask
- **Simplifies polygons**: Uses the Ramer-Douglas-Peucker algorithm to reduce point count while maintaining shape accuracy

## Files Modified/Created

### 1. New Action Implementation
**File**: `cvat-core/src/annotations-actions/masks-to-polygons.ts`

This file implements the `MasksToPolygons` class that:
- Extends `BaseShapesAction` to integrate with CVAT's action system
- Decodes RLE (Run-Length Encoding) mask data
- Finds contours using a marching squares algorithm
- Simplifies polygons to reduce point density
- Creates new polygon shapes from mask shapes

**Key Methods**:
- `run()`: Main entry point that processes all selected masks
- `convertMaskToPolygon()`: Converts individual mask to polygon
- `rleToPolygon()`: Decodes RLE mask data to polygon points
- `findContours()`: Finds boundaries in the binary mask
- `simplifyPolygon()`: Reduces polygon complexity using Douglas-Peucker algorithm

### 2. Action Registration
**File**: `cvat-core/src/annotations-actions/annotations-actions.ts`

Added:
```typescript
import { MasksToPolygons } from './masks-to-polygons';
registerAction(new MasksToPolygons());
```

This registers the action so it appears in the "Run actions" menu.

## How to Use

### Via Menu (Bulk Conversion)
1. Open an annotation task in CVAT
2. Click **Menu** (top left)
3. Select **Run actions**
4. Choose "Masks to polygons" from the dropdown
5. (Optional) Adjust the "simplificationTolerance" parameter:
   - Lower values (0.5-1.0): More detailed polygons with more points
   - Higher values (2.0-10.0): Simpler polygons with fewer points
   - Default: 1.5 (good balance)
6. Select frame range (or use "All frames", "Current frame", etc.)
7. Click **Run**

The converter will:
- Find all mask annotations in the selected range
- Convert each mask to a polygon
- Delete the original masks
- Create new editable polygons

### On Individual Objects
1. Right-click on a mask annotation
2. Select "Run action" from context menu
3. Choose "Masks to polygons"
4. Click **Run**

This converts only the selected mask.

## Technical Details

### RLE Mask Format
Masks in CVAT are stored in RLE (Run-Length Encoding) format:
```
[left, top, width, height, rle_data...]
```
- `left, top`: Bounding box position
- `width, height`: Bounding box dimensions
- `rle_data`: Alternating counts of 0s and 1s

### Conversion Algorithm

1. **Decode RLE** → Binary mask (2D array of 0s and 1s)
2. **Find Contours** → Trace boundaries using marching squares
3. **Select Largest** → Use the outer boundary (largest area)
4. **Convert to Absolute** → Transform local to global coordinates
5. **Simplify** → Reduce points using Douglas-Peucker
6. **Create Polygon** → Generate new polygon shape

### Marching Squares Algorithm
The contour tracing uses an 8-directional marching squares variant:
- Scans the binary mask for boundary pixels
- Traces the contour by following adjacent boundary pixels
- Returns a sequence of (x, y) coordinates

### Douglas-Peucker Simplification
Reduces polygon complexity while maintaining shape:
- Recursively removes points that contribute little to the shape
- Controlled by `simplificationTolerance` parameter
- Higher tolerance = fewer points = simpler polygon

## Parameters

### simplificationTolerance
- **Type**: Number (0.5 to 10.0)
- **Default**: 1.5
- **Effect**:
  - Controls how aggressively the polygon is simplified
  - Lower = more accurate but more points to edit
  - Higher = less accurate but easier to edit

## Limitations

1. **Small masks**: Very small masks (< 3 pixels) may not convert properly
2. **Complex shapes**: Highly detailed masks with many holes may be simplified
3. **Performance**: Large masks (1000s of pixels) may take a few seconds to convert
4. **Holes**: Only the outer boundary is extracted; inner holes are filled

## Troubleshooting

### "No masks found to convert"
- Ensure you have mask annotations in the selected frame range
- Check that filters aren't excluding your masks

### Polygon looks too simple
- Decrease the `simplificationTolerance` parameter
- Try values like 0.5 or 1.0 for more detail

### Polygon has too many points
- Increase the `simplificationTolerance` parameter
- Try values like 2.0 to 5.0 to reduce points

### Conversion fails silently
- Check browser console for error messages
- The mask may be corrupted or in an unexpected format

## Building the Code

After making changes to the TypeScript files:

```bash
# From the repository root
npm install        # Install dependencies (first time only)
npm run build      # Build all packages

# Or build just cvat-core
cd cvat-core
npm run build
```

Then restart the CVAT Docker containers to see the changes:

```bash
docker compose restart cvat_server cvat_ui
```

## Future Enhancements

Possible improvements for future versions:

1. **Preserve holes**: Extract and convert inner boundaries (holes)
2. **Multiple contours**: Convert each disconnected region to separate polygons
3. **Batch processing**: Add progress indicators for very large datasets
4. **Undo support**: Make conversion reversible
5. **Preview mode**: Show preview before committing conversion
6. **Smart simplification**: Adaptive tolerance based on mask complexity

## References

- Douglas-Peucker: https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
- Marching Squares: https://en.wikipedia.org/wiki/Marching_squares
- RLE Encoding: https://en.wikipedia.org/wiki/Run-length_encoding