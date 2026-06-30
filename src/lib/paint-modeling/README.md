# Paint Modeling

Paint modeling stores brushstrokes as continuous ribbon surfaces. A stroke owns its geometry. Charts, surface references, chart depth, chart raycast, chart overlays, and occlusion claims are not part of the current model.

## Core State

A saved view is still a calibrated camera. It records the camera matrices and viewport size used when the stroke was drawn.

A paint object owns strokes through `PaintStroke.objectId`. It no longer owns charts.

A paint stroke stores:

```ts
type PaintStroke = {
    id: string,
    objectId: string,
    sourceViewId: string,
    layerId: string,
    sourcePoints: Vec2[],
    centerline: Vec3[],
    mesh: {
        rows: number,
        columns: number[],
        closed: boolean,
        vertices: { position: Vec3, u: number, v: number }[],
        faces: [number, number, number, number][],
    },
    deformationLines: { id: string, points: { u: number, v: number }[] }[],
};
```

`u` runs along the stroke. `v` runs across the stroke width. The initial ribbon columns are `[-1, 0, 1]`.

## Stroke Commit

On pointer up, the draft path is sampled into a 2D source path. The source view projects each sample into free space at the active interaction depth. The stroke builder creates a ribbon frame across the stroke width and emits a row of shared vertices for each centerline sample.

Adjacent rows share quad faces:

```text
row i:     left_i   center_i   right_i
row i + 1: left_j   center_j   right_j

faces: [left_i, left_j, center_j, center_i]
       [center_i, center_j, right_j, right_i]
```

Closed source paths wrap the last row back to the first row. This makes ring-like strokes one connected surface, not many chart fragments.

## Rendering

Committed ribbons render from `stroke.mesh.faces`. Each quad emits two `RenderTriangle` primitives. Draft strokes use the same ribbon builder, but they are uploaded through the draft render path.

There is no chart upload store. There are no chart wire or surface-field primitives.

## Picking

Picking raycasts the actual stroke mesh. A hit returns:

```ts
type StrokeSurfaceHit = {
    objectId: string,
    strokeId: string,
    faceIndex: number,
    uv: { u: number, v: number },
    world: Vec3,
    viewDepth: number,
};
```

Future tools should use this hit instead of rebuilding chart coordinates.

## Sculpting

Depth sculpting is direct ribbon deformation. The sculpt operation moves nearby ribbon vertices with falloff in `(u, v)` space and preserves the face list. This keeps the surface connected while changing its shape.

## Deformation Lines

A deformation line is a purpose-bearing support line in ribbon coordinates. Inserting one adds local support columns around the line's `v` value and regenerates connected faces. The current implementation treats deformation lines as smooth support loops. It refines across width without increasing resolution everywhere along a separate 2D chart grid.

## Current Boundaries

View-claim locking is deferred. Strokes remember their source view, but up to three claimed views are not implemented yet.

Billboard paint is not a committed surface mode. The renderer keeps generic stroke primitives for guide and legacy render paths, but committed paint strokes are ribbons.