"""Exact filled-contour correspondence and finite point/body progress geometry."""
from __future__ import annotations

from pydantic import Field
from agents.yf_arc3_v5.capabilities.components import detect_components
from agents.yf_arc3_v5.capabilities.contracts import ComponentExtractionInput, FrameGrid
from agents.yf_arc3_v5.capabilities.local_field_pair_geometry import FootprintRow, footprint_distance_squared
from agents.yf_arc3_v5.logos.types import FrozenMap, FrozenModel, Ref


class FilledContourRequest(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.filled_contour_request.v1"
    before_frame: FrameGrid
    current_frame: FrameGrid
    before_point: tuple[int,int]
    current_component_refs: tuple[Ref,...] = Field(max_length=64)
    maximum_stencil_pixels: int = Field(ge=1,le=4096)
    current_occluder_bboxes: tuple[tuple[int,int,int,int],...] = Field(default=(),max_length=64)
    recalled_current_bbox: tuple[int,int,int,int] | None = None
    minimum_visible_fraction_numerator: int = Field(default=1,ge=1)
    minimum_visible_fraction_denominator: int = Field(default=1,ge=1)


class FilledContourMeasurements(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.filled_contour_measurements.v1"
    enumeration_complete: bool
    before_component_count: int
    current_match_rows: tuple[FrozenMap,...]


def filled_contour(component):
    """Fill enclosed holes only; preserve open concavities and the outer contour."""
    pixels=set(component.relative_pixels)
    height,width=component.bbox.height,component.bbox.width
    pending=[(r,c) for r in range(-1,height+1) for c in range(-1,width+1)
        if r in (-1,height) or c in (-1,width)]
    outside=set(pending)
    while pending:
        r,c=pending.pop()
        for dr,dc in ((-1,0),(0,-1),(0,1),(1,0)):
            q=(r+dr,c+dc)
            if -1<=q[0]<=height and -1<=q[1]<=width and q not in pixels and q not in outside:
                outside.add(q);pending.append(q)
    return tuple((r,c) for r in range(height) for c in range(width) if (r,c) not in outside)


def measure_filled_contour_correspondence(request: FilledContourRequest) -> FilledContourMeasurements:
    prior=[]
    for component in detect_components(ComponentExtractionInput(frame=request.before_frame,connectivity=4)).components:
        if component.touches_frame_boundary or component.area<=1:
            continue
        if component.bbox.height*component.bbox.width>request.maximum_stencil_pixels:
            return FilledContourMeasurements(enumeration_complete=False,before_component_count=0,current_match_rows=())
        shape=filled_contour(component)
        p=(request.before_point[0]-component.bbox.top,request.before_point[1]-component.bbox.left)
        if p in shape: prior.append((component,shape))
    rows=[]
    for current in detect_components(ComponentExtractionInput(frame=request.current_frame,connectivity=4)).components:
        if current.component_id not in request.current_component_refs: continue
        if current.bbox.height*current.bbox.width>request.maximum_stencil_pixels:
            return FilledContourMeasurements(enumeration_complete=False,before_component_count=len(prior),current_match_rows=())
        shape=filled_contour(current)
        for previous,old_shape in prior:
            placements={(current.bbox.top,current.bbox.left,current.bbox.bottom,current.bbox.right)}
            if request.recalled_current_bbox is not None: placements.add(request.recalled_current_bbox)
            for bbox in sorted(placements):
                top,left,bottom,right=bbox
                if bottom-top+1 != previous.bbox.height or right-left+1 != previous.bbox.width: continue
                expected={(top+r,left+c) for r,c in old_shape}
                observed={(current.bbox.top+r,current.bbox.left+c) for r,c in shape}
                def visible(p):
                    return not any(a<=p[0]<=c and b<=p[1]<=d for a,b,c,d in request.current_occluder_bboxes)
                expected_visible={p for p in expected if visible(p)}
                observed_visible={p for p in observed if visible(p)}
                if (len(expected_visible)*request.minimum_visible_fraction_denominator
                    < len(expected)*request.minimum_visible_fraction_numerator
                    or expected_visible!=observed_visible): continue
                rows.append(FrozenMap({"component_ref":current.component_id,
                    "before_component_ref":previous.component_id,"bbox":bbox,
                    "center":((top+bottom)//2,(left+right)//2),
                    "filled_contour_pixel_count":len(old_shape),"visible_corresponding_pixel_count":len(expected_visible),
                    "outer_contour_corresponds":True,"correspondence_under_body_occlusion":len(expected_visible)<len(expected)}))
    return FilledContourMeasurements(enumeration_complete=True,before_component_count=len(prior),current_match_rows=tuple(rows))


class PointProgressRequest(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.point_progress_request.v1"
    body: FootprintRow
    excluded_rows: tuple[FootprintRow,...] = Field(max_length=64)
    region_bbox: tuple[int,int,int,int]
    supplied_point: tuple[int,int]
    radius: int = Field(ge=1,le=64)
    maximum_point_checks: int = Field(ge=1,le=1048576)


class PointProgressMeasurements(FrozenModel):
    schema_version: Ref = "yf_arc3_v5.point_progress_measurements.v1"
    enumeration_complete: bool
    point_check_count: int
    candidate_rows: tuple[FrozenMap,...]


def measure_point_progress(request: PointProgressRequest) -> PointProgressMeasurements:
    top,left,bottom,right=request.region_bbox
    center2=(request.body.bbox[0]+request.body.bbox[2],request.body.bbox[1]+request.body.bbox[3])
    initial=sum((x-2*y)**2 for x,y in zip(center2,request.supplied_point))
    rows=[];checks=0
    for r in range(top,bottom+1):
        for c in range(left,right+1):
            checks+=1
            if checks>request.maximum_point_checks:
                return PointProgressMeasurements(enumeration_complete=False,point_check_count=checks,candidate_rows=())
            point=(r,c)
            residual=4*sum((x-y)**2 for x,y in zip(point,request.supplied_point))
            if residual>=initial or footprint_distance_squared(request.body.bbox,point)>=request.radius**2:
                continue
            if any(footprint_distance_squared(row.bbox,point)<=request.radius**2 for row in request.excluded_rows):
                continue
            rows.append(FrozenMap({"candidate_ref":f"point-progress:{request.body.item_ref}:{r}:{c}",
                "body_ref":request.body.item_ref,"point":point,"row":r,"col":c,
                "point_residual_squared_x4":residual,"current_residual_squared_x4":initial,
                "movement_squared_x4":sum((x-2*y)**2 for x,y in zip(center2,point)),"strict_domain_under_supplied_radius":True}))
    return PointProgressMeasurements(enumeration_complete=True,point_check_count=checks,candidate_rows=tuple(rows))
