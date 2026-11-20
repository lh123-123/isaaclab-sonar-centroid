// ROV competition universal parallel gripper
// Units: mm. This model is for fit/layout review; manufacture from the DXF/SVG.
$fn = 64;

base_w = 170;
base_h = 160;
base_t = 8;
jaw_len = 145;
jaw_w = 24;
jaw_t = 8;
opening = 120; // Set 0..120 to preview travel.

module hole(d, h=20) {
    cylinder(d=d, h=h, center=true);
}

module base_plate() {
    difference() {
        translate([-base_w/2, -base_h/2, 0])
            cube([base_w, base_h, base_t]);
        for (x=[-70,70], y=[-65,65])
            translate([x,y,base_t/2]) hole(6.5);
        for (x=[-49,59], y=[-32,32])
            translate([x,y,base_t/2]) hole(8.2);
        hole(16);
        for (a=[0:90:270])
            translate([25*cos(a),25*sin(a),base_t/2]) hole(5.5);
    }
}

module jaw() {
    difference() {
        cube([jaw_len,jaw_w,jaw_t]);
        for (x=[12,62,112])
            translate([x,jaw_w/2,jaw_t/2]) hole(4.2);
    }
}

module assembly() {
    color("silver") base_plate();
    // Rails.
    for (y=[-32,32])
        color("gray") translate([-49,y,12])
            rotate([0,90,0]) cylinder(d=8,h=145);
    // Fixed and translating jaw.
    color("steelblue") translate([-70,-72,20]) jaw();
    color("steelblue") translate([-70,-72+24+opening,20]) jaw();
}

assembly();
