import math


def binomial(successes, total):
    if total < 0 or not 0 <= successes <= total:
        raise ValueError('Invalid binomial counts')
    if total == 0:
        return dict(successes=0, total=0, rate=None, ci95=[None, None])
    p, z = successes / total, 1.959963984540054
    denominator = 1 + z*z / total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return dict(successes=successes, total=total, rate=p,
                ci95=[max(0., center-radius), min(1., center+radius)])


def summarize(rows):
    if not rows:
        raise ValueError('Cannot summarize empty predictions')
    n = len(rows)
    eligible = [r for r in rows if r['source_clean'] == r['label'] and r['target_clean'] == r['label']]
    source_success = [r for r in eligible if r['source_adv'] != r['label']]
    source_clean = [r for r in rows if r['source_clean'] == r['label']]
    return dict(
        n=n,
        source_clean_accuracy=binomial(sum(r['source_clean'] == r['label'] for r in rows), n),
        target_clean_accuracy=binomial(sum(r['target_clean'] == r['label'] for r in rows), n),
        adversarial_accuracy=binomial(sum(r['target_adv'] == r['label'] for r in rows), n),
        source_asr=binomial(sum(r['source_adv'] != r['label'] for r in source_clean), len(source_clean)),
        pair_transfer=binomial(sum(r['target_adv'] != r['label'] for r in eligible), len(eligible)),
        conditional_transfer=binomial(sum(r['target_adv'] != r['label'] for r in source_success), len(source_success)),
        mean_l2=sum(r['l2'] for r in rows)/n,
        max_linf=max(r['linf'] for r in rows))
