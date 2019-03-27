import json


if __name__ == "__main__":

    idx_mapping = {**{idx1: idx2 for idx1, idx2 in zip(range(53), range(80, 133))}, **{idx1: idx2 for idx1, idx2 in zip(range(53, 133), range(80))}}
    inv_idx_mapping = {**{idx1: idx2 for idx1, idx2 in zip(range(80), range(53, 133))}, **{idx1: idx2 for idx1, idx2 in zip(range(80, 133), range(53))}}
    for s in ['train', 'val']:

        pano_json = json.load(open('data/coco/annotations/panoptic_{}2017.json'.format(s)))

        cat_idx_mapping = {}
        for idx, k in enumerate(pano_json['categories']):
            cat_idx_mapping[k['id']] = idx

        pano_json_stff = pano_json.copy()
        for k, v in idx_mapping.items():      
            pano_json_stff['categories'][k] = pano_json['categories'][v]
            pano_json_stff['categories'][k]['id'] = k
        for anno in pano_json_stff['annotations']:
            for segments_info in anno['segments_info']:
                segments_info['category_id'] = inv_idx_mapping[cat_idx_mapping[segments_info['category_id']]]
        
        json.dump(pano_json_stff, open('data/coco/annotations/panoptic_{}2017_stff.json'.format(s), 'w'))

    json.dump(pano_json_stff['categories'], open('data/coco/annotations/panoptic_coco_categories_stff.json', 'w'))

