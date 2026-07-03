# Paint Modeling

Paint modeling stores brushstrokes as continuous ribbon surfaces. A stroke owns a sparse ribbon source. Charts, surface references, chart depth, chart raycast, chart overlays, occlusion claims, CPU ribbon raycast, and CPU evaluated ribbon meshes are not part of the current model.

## Core State

A saved view is a calibrated camera. It records the camera matrices and viewport size used when the stroke was drawn.

A paint object owns strokes through `PaintStroke.objectId`. It no longer owns charts.

A paint stroke stores:

```ts
type PaintStroke = {
    id: string,
    objectId: string,
    sourceViewId: string,
    layerId: string,
    sourcePoints: Vec2[],
    ribbon: {
        closed: boolean,
        vertices: {
            position: Vec3,
            side: Vec3,
            u: number,
        }[],
    },
};
```

`u` runs along the stroke. Each ribbon vertex is oriented: `position` is the center point and `side` is the half-width vector across the ribbon.

## Stroke Commit

On pointer up, the draft path is sampled into a 2D source path. The source view projects each sample into free space at the active interaction depth. The stroke builder emits one oriented vertex per centerline sample.

Closed source paths drop the duplicated closing source point and mark the ribbon as closed. The GPU wraps the last oriented vertex back to the first one when it expands the ribbon.

## Rendering

Committed and draft ribbons render as `RenderRibbon` primitives. They upload only oriented vertices plus color and shading uniforms.

The ribbon shader expands adjacent oriented vertices into the two ribbon edges and triangles with `vertex_index`. CPU code does not build ribbon faces, evaluated surfaces, render triangles, or raycast triangles. Draft ribbons render opaque and use a small GPU clip-depth bias so preview strokes do not fight committed surfaces.

There is no chart upload store. There are no chart wire or surface-field primitives.

## Picking

There is no committed CPU ribbon picking path. Future picking should use a GPU-backed path or a deliberate source-level interaction model, not a CPU evaluated surface.

## Current Boundaries

View-claim locking is deferred. Strokes remember their source view, but up to three claimed views are not implemented yet.

Billboard paint is not a committed surface mode. The renderer keeps generic stroke primitives for guide and legacy render paths, but committed paint strokes are ribbons.
